# motion-detection
detect motionflow in video with opencv optical flow and object detection


# requirements
opencv4

# command
  g++ -o motiondetect motiondetection.cpp $(pkg-config opencv4 --libs --cflags)
  ./motiondetect
