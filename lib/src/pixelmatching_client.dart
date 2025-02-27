import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as imglib;
import '../flutter_pixelmatching.dart';
import 'pixelmatching_bindings.dart' as bindings;

final DynamicLibrary _lib = Platform.isAndroid
    ? DynamicLibrary.open("libflutter_pixelmatching.so")
    : DynamicLibrary.process();

final _native = bindings.PixelMatchingBindings(_lib);

late _PixelMatchingClient _client;

late SendPort _caller;

class Request {
  int id;
  String method;
  dynamic params;

  Request({
    required this.id,
    required this.method,
    this.params,
  });
}

class Response {
  int id;
  dynamic data;

  Response({
    required this.id,
    this.data,
  });
}

void init(
  SendPort caller,
) {
  _client = _PixelMatchingClient();
  _caller = caller;

  // 메인 스레드에 메시지를 보낼 포트를 저장합니다.
  final fromMainThread = ReceivePort();
  // 메인 스레드가 메시지를 보내고 수신할 수 있는 포트를 만듭니다.
  fromMainThread.listen(_handleMessage);
  // 메인 스레드에 메시지를 보낼 수 있는 포트를 보냅니다.
  _caller.send(fromMainThread.sendPort);
}

// 클라이언트로 요청온 데이터들을 처리합니다.
void _handleMessage(message) {
  if (message is Request) {
    // initialize
    switch (message.method) {
      case "initialize":
        _client.initialize();
        var status = _client.getStateCode();
        // initialize
        if (status != PixelMatchingState.noMarker) {
          _caller.send(Response(id: message.id, data: false));
          return;
        }
        // set target image
        final imageType = message.params['imageType'] as String;
        final image = message.params['image'] as Uint8List;
        final w = message.params['width'] as int;
        final h = message.params['height'] as int;
        final rotation = message.params['rotation'] as int;
        final res = _client.setTargetImageFromBytes(imageType, image, w, h,
            rotation: rotation);
        _caller.send(Response(id: message.id, data: res));
        break;
      case "getStateCode":
        var status = _client.getStateCode();
        _caller.send(Response(id: message.id, data: status));
        break;
      case "getMarkerQueryDifferenceImage":
        final res = _client.getMarkerQueryDifferenceImage();
        _caller.send(Response(id: message.id, data: res));
        break;
      case "setQuery":
        if (message.params is Map<String, dynamic>) {
          final imageType = message.params['imageType'] as String;
          final image = message.params['image'] as Uint8List;
          final width = message.params['width'] as int;
          final height = message.params['height'] as int;
          final rotation = message.params['rotation'] as int;
          final res = _client.query(imageType, image, width, height,
              rotation: rotation);
          _caller.send(Response(id: message.id, data: res));
        }
        break;
      case "cameraImageToQuery":
        if (message.params is CameraImage) {
          final image = message.params as CameraImage;
          final res = cameraImageToQuery(_native, image);
          _caller.send(Response(id: message.id, data: res));
        }
        break;
      case "dispose":
        _client.dispose();
        _caller.send(Response(id: message.id));
        break;
    }
  }
}

class _PixelMatchingClient {
  var _imageBufferSize = 0;
  var _imageTypeString = "";

  // Image Buffer
  Pointer<Uint8>? _imageBuffer;
  Pointer<Utf8>? _imageType;
  Pointer<Int>? _debugImageLen;

  void initialize() {
    _native.initialize();
  }

  bool setTargetImageFromBytes(String imageType, Uint8List bytes, int w, int h,
      {int rotation = 0}) {
    var totalSize = bytes.length;
    var imgBuffer = malloc.allocate<Uint8>(totalSize);
    var imgBufferList = imgBuffer.asTypedList(totalSize);
    var imgType = imageType.toNativeUtf8();
    imgBufferList.setAll(0, bytes);
    var res =
        _native.setMarker(imgType.cast(), imgBuffer.cast(), w, h, rotation);
    malloc.free(imgBuffer);
    malloc.free(imgType);
    return res;
  }

  String query(String imageType, Uint8List bytes, int w, int h,
      {int rotation = 0}) {
    if (_imageBuffer != null && _imageBufferSize != bytes.length) {
      malloc.free(_imageBuffer!);
      _imageBuffer = null;
    }
    if (_imageType != null && imageType != _imageTypeString) {
      malloc.free(_imageType!);
      _imageType = null;
    }
    _imageBufferSize = bytes.length;
    _imageBuffer ??= malloc.allocate<Uint8>(_imageBufferSize);
    _imageType ??= imageType.toNativeUtf8();
    _imageTypeString = imageType;

    // 이미지 내용 복사
    var imageBufferList = _imageBuffer!.asTypedList(_imageBufferSize);
    imageBufferList.setAll(0, bytes);

    var res = _native.setQuery(
        _imageType!.cast(), _imageBuffer!.cast(), w, h, rotation);
    if (res) {
      return _confidence();
    } else {
      return "";
    }
  }

  imglib.Image? getMarkerQueryDifferenceImage() {
    _debugImageLen ??= malloc.allocate<Int>(1);
    _debugImageLen?.value = 0;
    final imgPointer = _native.getMarkerQueryDifferenceImage(_debugImageLen!);
    if (imgPointer.address == nullptr.address) {
      return null;
    }
    final imgLen = _debugImageLen!.value;
    if (imgLen == 0) {
      malloc.free(imgPointer);
      return null;
    }
    final imgBytes = imgPointer.cast<Uint8>().asTypedList(imgLen);
    final img = imglib.decodeJpg(imgBytes);
    malloc.free(imgPointer);
    return img;
  }

  PixelMatchingState getStateCode() {
    final state = _native.getStateCode();
    if (state == -1) {
      return PixelMatchingState.notInitialized;
    } else if (state == 0) {
      return PixelMatchingState.noMarker;
    } else if (state == 1) {
      return PixelMatchingState.noQuery;
    } else if (state == 2) {
      return PixelMatchingState.processing;
    } else {
      throw Exception("Unknown state");
    }
  }

  // if it comes out to a 0.87, it can be judged as a matching
  String _confidence() {
    return _native.getQueryConfidenceRate();
  }

  void dispose() {
    if (_imageBuffer != null) {
      malloc.free(_imageBuffer!);
    }
    if (_debugImageLen != null) {
      malloc.free(_debugImageLen!);
    }
    if (_imageType != null) {
      malloc.free(_imageType!);
    }
    _imageBuffer = null;
    _native.dispose();
  }
}
