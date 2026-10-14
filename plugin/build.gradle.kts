plugins {
    java
}

group = "dev.mujocraft"
version = "0.1.0"

repositories {
    mavenCentral()
    maven("https://repo.papermc.io/repository/maven-public/")
}

dependencies {
    // Paper 26.3 build 181 (stable). The repo's version strings carry a channel suffix.
    compileOnly("io.papermc.paper:paper-api:26.3.build.181-stable")
}

java {
    toolchain.languageVersion.set(JavaLanguageVersion.of(25))
}

// The source tree lives in OneDrive, which locks files Gradle needs to delete/regenerate
// ("Unable to delete directory ... build\generated"). Build outside the synced folder.
layout.buildDirectory.set(
    file(System.getenv("LOCALAPPDATA") ?: System.getProperty("java.io.tmpdir"))
        .resolve("mujocraft-plugin-build")
)
