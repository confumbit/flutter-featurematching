mkdir -p download
cd download

wget -O opencv-4.5.4-android-sdk.zip https://sourceforge.net/projects/opencvlibrary/files/4.5.4/opencv-4.5.4-android-sdk.zip/download
wget -O opencv-4.5.4-ios-framework.zip https://sourceforge.net/projects/opencvlibrary/files/4.5.4/opencv-4.5.4-ios-framework.zip/download

unzip opencv-4.5.4-android-sdk.zip
unzip opencv-4.5.4-ios-framework.zip

cp -r opencv2.framework ../ios
cp -r OpenCV-android-sdk/sdk/native/jni/include ../
mkdir -p ../android/src/main/cmakeLibs/
cp -r OpenCV-android-sdk/sdk/native/libs/* ../android/src/main/cmakeLibs/
cd ..
