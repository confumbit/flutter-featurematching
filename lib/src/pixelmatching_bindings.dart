// AUTO GENERATED FILE, DO NOT EDIT.
//
// Generated by `package:ffigen`.
// ignore_for_file: type=lint
import 'dart:ffi' as ffi;

/// FFI bindings for OpenCV
class PixelMatchingBindings {
  /// Holds the symbol lookup function.
  final ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
      _lookup;

  /// The symbols are looked up in [dynamicLibrary].
  PixelMatchingBindings(ffi.DynamicLibrary dynamicLibrary)
      : _lookup = dynamicLibrary.lookup;

  /// The symbols are looked up with [lookup].
  PixelMatchingBindings.fromLookup(
      ffi.Pointer<T> Function<T extends ffi.NativeType>(String symbolName)
          lookup)
      : _lookup = lookup;

  /// @return initialize result
  bool initialize() {
    return _initialize();
  }

  late final _initializePtr =
      _lookup<ffi.NativeFunction<ffi.Bool Function()>>('initialize');
  late final _initialize = _initializePtr.asFunction<bool Function()>();

  /// @return state code
  int getStateCode() {
    return _getStateCode();
  }

  late final _getStateCodePtr =
      _lookup<ffi.NativeFunction<ffi.Int Function()>>('getStateCode');
  late final _getStateCode = _getStateCodePtr.asFunction<int Function()>();

  /// @return set marker result
  bool setMarker(
    ffi.Pointer<ffi.UnsignedChar> image,
    int width,
    int height,
    int rotation,
  ) {
    return _setMarker(
      image,
      width,
      height,
      rotation,
    );
  }

  late final _setMarkerPtr = _lookup<
      ffi.NativeFunction<
          ffi.Bool Function(ffi.Pointer<ffi.UnsignedChar>, ffi.Int, ffi.Int,
              ffi.Int)>>('setMarker');
  late final _setMarker = _setMarkerPtr.asFunction<
      bool Function(ffi.Pointer<ffi.UnsignedChar>, int, int, int)>();

  /// @return set query result
  bool setQuery(
    ffi.Pointer<ffi.UnsignedChar> image,
    int width,
    int height,
    int rotation,
  ) {
    return _setQuery(
      image,
      width,
      height,
      rotation,
    );
  }

  late final _setQueryPtr = _lookup<
      ffi.NativeFunction<
          ffi.Bool Function(ffi.Pointer<ffi.UnsignedChar>, ffi.Int, ffi.Int,
              ffi.Int)>>('setQuery');
  late final _setQuery = _setQueryPtr.asFunction<
      bool Function(ffi.Pointer<ffi.UnsignedChar>, int, int, int)>();

  /// match marker and query
  /// @return match confidence rate
  double getQueryConfidenceRate() {
    return _getQueryConfidenceRate();
  }

  late final _getQueryConfidenceRatePtr =
      _lookup<ffi.NativeFunction<ffi.Double Function()>>(
          'getQueryConfidenceRate');
  late final _getQueryConfidenceRate =
      _getQueryConfidenceRatePtr.asFunction<double Function()>();

  /// @param size output image size
  /// @return output image bytes(encoded by jpeg)
  ffi.Pointer<ffi.UnsignedChar> getMarkerQueryDifferenceImage(
    ffi.Pointer<ffi.Int> size,
  ) {
    return _getMarkerQueryDifferenceImage(
      size,
    );
  }

  late final _getMarkerQueryDifferenceImagePtr = _lookup<
      ffi.NativeFunction<
          ffi.Pointer<ffi.UnsignedChar> Function(
              ffi.Pointer<ffi.Int>)>>('getMarkerQueryDifferenceImage');
  late final _getMarkerQueryDifferenceImage =
      _getMarkerQueryDifferenceImagePtr.asFunction<
          ffi.Pointer<ffi.UnsignedChar> Function(ffi.Pointer<ffi.Int>)>();

  void dispose() {
    return _dispose();
  }

  late final _disposePtr =
      _lookup<ffi.NativeFunction<ffi.Void Function()>>('dispose');
  late final _dispose = _disposePtr.asFunction<void Function()>();
}

const int __bool_true_false_are_defined = 1;

const int true1 = 1;

const int false1 = 0;
