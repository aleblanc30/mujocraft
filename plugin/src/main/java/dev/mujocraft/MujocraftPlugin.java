package dev.mujocraft;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.BlockDisplay;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Transformation;
import org.joml.Quaternionf;
import org.joml.Vector3f;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Bridge between a Minecraft world and an external MuJoCo process (see python/ in this repo).
 *
 * Reads solidity of blocks on request, forwards block place/break deltas, and renders the
 * simulated agent (which stands in for the player) as one BlockDisplay per MuJoCo box geom.
 *
 * Wire format is documented in src/mujocraft/protocol.py.
 */
public final class MujocraftPlugin extends JavaPlugin implements Listener {
    private static final Gson GSON = new Gson();
    private static final int PROTOCOL_VERSION = 1;

    private volatile boolean running = true;
    private volatile ServerSocket server;
    private volatile BufferedWriter out;
    private volatile JsonObject latestPose;

    // main-thread state
    private World world;
    private final Map<String, BlockDisplay> displays = new HashMap<>();
    private final List<int[]> pendingChanges = new ArrayList<>();
    private int tickCount = 0;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        Bukkit.getPluginManager().registerEvents(this, this);
        Bukkit.getScheduler().runTaskTimer(this, this::tick, 1L, 1L);
        Thread t = new Thread(this::acceptLoop, "mujocraft-accept");
        t.setDaemon(true);
        t.start();
    }

    @Override
    public void onDisable() {
        running = false;
        try {
            if (server != null) server.close();
        } catch (IOException ignored) {
        }
        clearDisplays();
    }

    // ---- networking (off main thread) ---------------------------------------------------------

    private void acceptLoop() {
        try {
            String bind = getConfig().getString("bind", "127.0.0.1");
            int port = getConfig().getInt("port", 25566);
            server = new ServerSocket(port, 1, InetAddress.getByName(bind));
            getLogger().info("Waiting for MuJoCo bridge on " + bind + ":" + port);
            while (running) {
                try (Socket s = server.accept()) {
                    s.setTcpNoDelay(true);
                    serve(s);
                } catch (IOException e) {
                    if (running) getLogger().warning("Bridge connection ended: " + e.getMessage());
                }
                out = null;
                latestPose = null;
                Bukkit.getScheduler().runTask(this, this::clearDisplays);
            }
        } catch (IOException e) {
            getLogger().severe("Could not open bridge socket: " + e.getMessage());
        }
    }

    private void serve(Socket s) throws IOException {
        out = new BufferedWriter(new OutputStreamWriter(s.getOutputStream(), StandardCharsets.UTF_8));
        BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.UTF_8));
        String line;
        while (running && (line = in.readLine()) != null) {
            JsonObject msg = GSON.fromJson(line, JsonObject.class);
            String type = msg.get("type").getAsString();
            if (type.equals("pose")) {
                latestPose = msg; // latest wins; applied once per tick
            } else {
                Bukkit.getScheduler().runTask(this, () -> handle(type, msg));
            }
        }
    }

    private void send(JsonObject msg) {
        BufferedWriter w = out;
        if (w == null) return;
        try {
            synchronized (w) {
                w.write(GSON.toJson(msg));
                w.write('\n');
                w.flush();
            }
        } catch (IOException e) {
            out = null;
        }
    }

    // ---- request handling (main thread) -------------------------------------------------------

    private void handle(String type, JsonObject msg) {
        switch (type) {
            case "hello" -> onHello();
            case "request_region" -> onRequestRegion(msg);
            default -> getLogger().fine("Ignoring message type " + type);
        }
    }

    private void onHello() {
        // No player is involved: the spider is the actor. Spawn comes from config, else world spawn.
        World w = Bukkit.getWorld(getConfig().getString("world", ""));
        if (w == null) w = Bukkit.getWorlds().get(0);
        world = w;
        double x, y, z;
        if (getConfig().isList("spawn") && getConfig().getDoubleList("spawn").size() == 3) {
            var s = getConfig().getDoubleList("spawn");
            x = s.get(0);
            y = s.get(1);
            z = s.get(2);
        } else {
            Location l = w.getSpawnLocation();
            x = l.getBlockX() + 0.5;
            z = l.getBlockZ() + 0.5;
            y = w.getHighestBlockYAt(l.getBlockX(), l.getBlockZ()) + 1.5;
        }
        JsonObject m = new JsonObject();
        m.addProperty("type", "welcome");
        m.addProperty("version", PROTOCOL_VERSION);
        m.add("spawn", vec(x, y, z));
        send(m);
    }

    private void onRequestRegion(JsonObject msg) {
        if (world == null) return;
        int[] o = ints(msg.getAsJsonArray("origin"));
        int[] n = ints(msg.getAsJsonArray("size"));
        byte[] bits = new byte[(n[0] * n[1] * n[2] + 7) / 8];
        int i = 0;
        for (int x = 0; x < n[0]; x++)
            for (int y = 0; y < n[1]; y++)
                for (int z = 0; z < n[2]; z++, i++)
                    if (world.getBlockAt(o[0] + x, o[1] + y, o[2] + z).getType().isSolid())
                        bits[i >> 3] |= (byte) (0x80 >> (i & 7));
        JsonObject m = new JsonObject();
        m.addProperty("type", "region");
        m.add("origin", msg.get("origin"));
        m.add("size", msg.get("size"));
        m.addProperty("solid", Base64.getEncoder().encodeToString(bits));
        send(m);
    }

    // ---- per-tick work ------------------------------------------------------------------------

    private void tick() {
        tickCount++;
        if (out == null) return;

        if (!pendingChanges.isEmpty()) {
            JsonArray arr = new JsonArray();
            for (int[] c : pendingChanges) {
                JsonArray e = new JsonArray();
                for (int v : c) e.add(v);
                arr.add(e);
            }
            pendingChanges.clear();
            JsonObject m = new JsonObject();
            m.addProperty("type", "blocks");
            m.add("changes", arr);
            send(m);
        }

        JsonObject pose = latestPose;
        if (pose != null && world != null) {
            latestPose = null;
            applyPose(pose);
        }
    }

    private void applyPose(JsonObject pose) {
        for (JsonElement el : pose.getAsJsonArray("geoms")) {
            JsonObject g = el.getAsJsonObject();
            String id = g.get("id").getAsString();
            float[] p = floats(g.getAsJsonArray("pos"));
            float[] q = floats(g.getAsJsonArray("quat")); // x y z w
            float[] s = floats(g.getAsJsonArray("size"));

            Location loc = new Location(world, p[0], p[1], p[2]);
            BlockDisplay d = displays.get(id);
            if (d == null || !d.isValid()) {
                String block = g.get("block").getAsString();
                d = world.spawn(loc, BlockDisplay.class, e -> {
                    e.setBlock(Bukkit.createBlockData(block));
                    e.setPersistent(false);
                    e.setInterpolationDuration(2);
                    e.setTeleportDuration(2);
                    e.setViewRange(4f);
                });
                displays.put(id, d);
            }

            // A BlockDisplay draws a unit cube from its local origin. Centre it on `loc`, then
            // rotate/scale to the MuJoCo box.
            Vector3f size = new Vector3f(s[0], s[1], s[2]);
            Quaternionf rot = new Quaternionf(q[0], q[1], q[2], q[3]).normalize();
            Vector3f t = new Vector3f(size).mul(0.5f).rotate(rot).negate();
            d.setTransformation(new Transformation(t, rot, size, new Quaternionf()));
            d.setInterpolationDelay(0);
            d.teleport(loc);
        }
    }

    private void clearDisplays() {
        for (BlockDisplay d : displays.values()) d.remove();
        displays.clear();
    }

    // ---- block deltas -------------------------------------------------------------------------

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onBreak(BlockBreakEvent e) {
        var b = e.getBlock();
        pendingChanges.add(new int[]{b.getX(), b.getY(), b.getZ(), 0});
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void onPlace(BlockPlaceEvent e) {
        var b = e.getBlockPlaced();
        pendingChanges.add(new int[]{b.getX(), b.getY(), b.getZ(), b.getType().isSolid() ? 1 : 0});
    }

    // ---- helpers ------------------------------------------------------------------------------

    private static JsonArray vec(double x, double y, double z) {
        JsonArray a = new JsonArray();
        a.add(x);
        a.add(y);
        a.add(z);
        return a;
    }

    private static int[] ints(JsonArray a) {
        int[] r = new int[a.size()];
        for (int i = 0; i < r.length; i++) r[i] = a.get(i).getAsInt();
        return r;
    }

    private static float[] floats(JsonArray a) {
        float[] r = new float[a.size()];
        for (int i = 0; i < r.length; i++) r[i] = a.get(i).getAsFloat();
        return r;
    }
}
