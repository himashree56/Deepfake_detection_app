# Keep TensorFlow Lite GPU Delegate and related classes
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegateFactory$Options { *; }
-keepclassmembers class * {
    @org.tensorflow.lite.annotations.* <fields>;
    @org.tensorflow.lite.annotations.* <methods>;
}
-dontwarn org.tensorflow.lite.**
