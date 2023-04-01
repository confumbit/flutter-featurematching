import 'dart:developer';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_pixelmatching/flutter_pixelmatching.dart';
import 'package:image/image.dart' as imglib;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  CameraController? controller;
  FlutterPixelMatching pixelmatching = FlutterPixelMatching();
  bool process = false;
  // capturing images
  imglib.Image? capture;
  CameraImage? cameraImage;
  bool isTarget = false;

  @override
  void initState() {
    super.initState();
    init();
  }

  init() async {
    pixelmatching.initialize();

    final cameras = await availableCameras();
    controller = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    controller!.initialize().then((_) {
      if (!mounted) {
        return;
      }
      controller!.startImageStream((CameraImage cameraImage) {
        cameraStream(cameraImage);
      });
      setState(() {});
    });
  }

  cameraStream(CameraImage cameraImage) {
    this.cameraImage = cameraImage;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('OpenCV PixelMatching Plugin Sample'),
        ),
        body: capture == null
            ? Column(
                children: [
                  controller != null ? CameraPreview(controller!) : Container(),
                ],
              )
            : Image.memory(imglib.encodeJpg(capture!)),
        floatingActionButton: IconButton(
          onPressed: () {
            if (process) return;
            process = true;
            // final h = Platform.isAndroid ? cameraImage.width : cameraImage.height;
            // final w = Platform.isAndroid ? cameraImage.height : cameraImage.width;
            // capture = pixelmatching.extractFeaturesAndEncodeToJpeg(cameraImage!);
            // if (capture != null) {
            //   setState(() {});
            // }
            if (isTarget == false) {
              pixelmatching.setTargetImage(cameraImage!);
              isTarget = true;
            } else {
              pixelmatching.setQueryImage(cameraImage!);
              final result = pixelmatching.getQueryConfidenceRate();
              log('[pixelmatching] result : $result');
            }
            process = false;
          },
          icon: Icon(
            Icons.camera,
          ),
        ),
      ),
    );
  }
}
