#!/bin/bash

# Repast Simphony Model run script for macOS and Linux

PWD="${0%/*}"
cd "$PWD"

# Note the Repast Simphony Directories.
REPAST_SIMPHONY_ROOT=$PWD/repast.simphony/repast.simphony.runtime_$REPAST_VERSION

# Define the Core Repast Simphony Directories and JARs
CP=$CP:$REPAST_SIMPHONY_ROOT/bin
CP=$CP:$REPAST_SIMPHONY_ROOT/lib/*

# Groovy jar
CP=$CP:$PWD/groovylib/$Groovy_Jar

# User model lib jars
CP=$CP:lib/*

# Change to the project directory
cd "camera"

# Start the Model
java -XX:+IgnoreUnrecognizedVMOptions \
	--add-opens java.base/java.lang.reflect=ALL-UNNAMED \
    --add-modules=ALL-SYSTEM \
    --add-exports=java.base/jdk.internal.ref=ALL-UNNAMED \
    --add-exports=java.desktop/sun.awt=ALL-UNNAMED \
    --add-exports=java.base/java.lang=ALL-UNNAMED \
    --add-opens java.base/java.util=ALL-UNNAMED \
    --add-exports=java.xml/com.sun.org.apache.xpath.internal.objects=ALL-UNNAMED \
    --add-exports=java.xml/com.sun.org.apache.xpath.internal=ALL-UNNAMED \
    --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "$CP" repast.simphony.runtime.RepastMain  "./camera.rs"