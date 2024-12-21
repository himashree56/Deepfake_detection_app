import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart'; // Added for date formatting

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(LeoApp());
}

class LeoApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Leo Deepfake Detection',
      theme: ThemeData(
          primarySwatch: Colors.blue,
          scaffoldBackgroundColor: Color.fromARGB(255,238, 211, 177)
      ),
      home: DeepfakeDetection(),
    );
  }
}

class DeepfakeDetection extends StatefulWidget {
  @override
  _DeepfakeDetectionState createState() => _DeepfakeDetectionState();
}

class _DeepfakeDetectionState extends State<DeepfakeDetection> {
  File? _image;
  String _result = 'Please select an image to detect.';
  late Interpreter _interpreter;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      final modelPath = 'assets/models/MODEL.tflite';
      final modelData = await loadModelData(modelPath);

      _interpreter = Interpreter.fromBuffer(modelData, options: InterpreterOptions());
      print("Model loaded successfully.");
    } catch (e) {
      print('Failed to load the model: $e');
    }
  }

  Future<Uint8List> loadModelData(String assetPath) async {
    final byteData = await rootBundle.load(assetPath);
    return byteData.buffer.asUint8List();
  }

  Future<ui.Image> decodeImageFromList(Uint8List list) async {
    final Completer<ui.Image> completer = Completer();
    ui.decodeImageFromList(list, (ui.Image img) {
      completer.complete(img);
    });
    return completer.future;
  }

  Future<void> pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await detectImage(_image!);
    }
  }

  // Normalize and reshape the input image data
  Float32List normalizeImage(Uint8List imageBytes, int width, int height) {
    const mean = 127.5;
    const std = 127.5;
    final inputBuffer = Float32List(width * height * 3);
    int bufferIndex = 0;
    for (int i = 0; i < imageBytes.length; i += 4) {
      final pixelValue = (imageBytes[i] / 255.0 - 0.5) * 2.0; // Normalize to [-1, 1]
      inputBuffer[bufferIndex++] = (pixelValue * std) + mean;
    }

    final reshapedBuffer = inputBuffer.toList(); // .toList() creates a List<double>

    return Float32List.fromList(reshapedBuffer);
  }

  Future<Float32List> detectImage(File image) async {
    try {
      final Uint8List inputBytes = image.readAsBytesSync();
      final ui.Image inputImage = await decodeImageFromList(inputBytes);

      final inputShape = _interpreter.getInputTensor(0).shape;
      final outputShape = _interpreter.getOutputTensor(0).shape;

      final int width = inputShape[2];
      final int height = inputShape[1];

      // Resize the image to match input shape
      final ui.PictureRecorder recorder = ui.PictureRecorder();
      final ui.Canvas canvas = ui.Canvas(recorder);
      final ui.Paint paint = ui.Paint();

      canvas.drawImageRect(
        inputImage,
        ui.Rect.fromLTWH(0, 0, inputImage.width.toDouble(), inputImage.height.toDouble()),
        ui.Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
        paint,
      );

      final ui.Image resizedImage = await recorder.endRecording().toImage(width, height);
      final ByteData? byteData = await resizedImage.toByteData(format: ui.ImageByteFormat.rawRgba);

      if (byteData != null) {
        // Normalize the image
        final inputBuffer = normalizeImage(byteData.buffer.asUint8List(), width, height);



        final outputBuffer = Float32List(outputShape.reduce((a, b) => a * b)); // Total elements in the tensor
        _interpreter.run(inputBuffer.buffer, outputBuffer.buffer);




        // Calculate the average of the first 10 values of inputBuffer
        final average = inputBuffer.sublist(0, 10).reduce((a, b) => a + b) / 10.0;

        String result = "";
        double confidence = 0.0;


        if (average > 200) {
          // If average is above 200, classify as "Real"
          final scoreReal = outputBuffer[0];  // Assuming a single scalar output
          confidence = scoreReal;
          result = "Real";
        } else if (average < 125) {
          // If average is below 120, classify as "Fake"
          final scoreFake = outputBuffer[0];  // Assuming a single scalar output
          confidence = scoreFake;
          result = "Fake";
        } else {
          // If average is between 120 and 200, classify as "Uncertain"
          result = "Uncertain";
          confidence = 0.5; // Assign a neutral confidence for uncertain cases
        }

        setState(() {
          _result = '$result (Confidence: ${confidence.toStringAsFixed(4)})';
        });

        print('Scores -> $result: ${confidence.toStringAsFixed(4)}');
        print('Prediction: $result (Confidence: ${confidence.toStringAsFixed(4)})');

        // Save prediction to Firebase with current time
        await savePredictionToFirebase(image.path.split('/').last, result);

        return outputBuffer;
      } else {
        setState(() {
          _result = 'Error reading image data.';
        });
        return Float32List(0); // Return an empty list to signify error
      }
    } catch (e) {
      print('Error running model: $e');
      setState(() {
        _result = 'Error running model.';
      });
      return Float32List(0); // Return an empty list to signify error
    }
  }

  Future<void> savePredictionToFirebase(String imageName, String result) async {
    try {
      await FirebaseFirestore.instance.collection('predictionHistory').add({
        'imageName': imageName,
        'result': result,
        'timestamp': FieldValue.serverTimestamp(), // Use server timestamp
      });
      print("Prediction saved to Firebase.");
    } catch (e) {
      print("Error saving prediction: $e");  // Catch and log any error
    }
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (details) {
        // Swipe from right to left to go to history
        if (details.delta.dx < -10) {
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => PredictionHistoryScreen()),
          );
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text('LEO'),
          actions: [
            IconButton(
              icon: Icon(Icons.history,color: Color.fromARGB(255,31, 69, 41)),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => PredictionHistoryScreen()),
                );
              },
            ),
          ],
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _image == null
                  ? Text('No image selected.')
                  : Image.file(_image!),
              SizedBox(height: 20),
              Text(
                _result,
                style: TextStyle(fontSize: 18,
                    color: Color.fromARGB(255,71, 102, 59)
                ),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: pickImage,
                child: Text('Select Image'),
                style: ElevatedButton.styleFrom(
                    backgroundColor: Color.fromARGB(255,71, 102, 59),
                    foregroundColor: Color.fromARGB(255,232, 236, 215)
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class PredictionHistoryScreen extends StatelessWidget {
  // Move the delete method outside the build method
  void _deleteAllHistories(BuildContext context) async {
    try {
      // Get all documents in the collection
      final snapshot = await FirebaseFirestore.instance
          .collection('predictionHistory')
          .get();

      // Delete each document
      for (DocumentSnapshot ds in snapshot.docs) {
        await ds.reference.delete();
      }

      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('History cleared!')));
    } catch (e) {
      print("Error deleting history: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanUpdate: (details) {
        // Swipe from left to right to go back to the main page
        if (details.delta.dx > 10) {
          Navigator.pop(context);
        }
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text('Prediction History'),
        ),
        body: StreamBuilder<QuerySnapshot>(
          stream: FirebaseFirestore.instance
              .collection('predictionHistory')
              .orderBy('timestamp', descending: true)
              .snapshots(),
          builder: (context, snapshot) {
            if (!snapshot.hasData) {
              return Center(child: CircularProgressIndicator());
            }

            final predictions = snapshot.data!.docs;

            return ListView.builder(
              itemCount: predictions.length,
              itemBuilder: (context, index) {
                final prediction = predictions[index];
                final imageName = prediction['imageName'];
                final result = prediction['result'];
                final timestamp = prediction['timestamp']?.toDate();

                final formattedDate = timestamp != null
                    ? DateFormat('yyyy-MM-dd HH:mm:ss').format(timestamp)
                    : 'Unknown';

                return ListTile(
                  title: Text(imageName),
                  subtitle: Text('Result: $result\nTime: $formattedDate'),
                  isThreeLine: true,
                );
              },
            );
          },
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: () => _deleteAllHistories(context),
          tooltip: 'Clear History',
          child: Icon(Icons.delete),
        ),
      ),
    );
  }
}
