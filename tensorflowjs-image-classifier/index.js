const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

let net;


async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({
          video: true
        },
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata', () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}


async function app() {
  let loadingMessage = 'Downloading MobileNet v2..'
  console.log(loadingMessage);
  document.getElementById('loading').innerText = `${loadingMessage}`;

  // Load the model.
  net = await mobilenet.load({version: 2, alpha: 1.0});
  loadingMessage = 'Sucessfully loaded model'
  console.log(loadingMessage);
  document.getElementById('loading').innerText = `${loadingMessage}`;

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    console.log(classId)
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  while (true) {
    // Show me what you actually see
    const result = await net.classify(webcamElement);

    document.getElementById('vision').innerText = `
      seeing now: ${result[0].className}\n
      probability: ${result[0].probability}
    `;

    // and now tell me what you classified
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;

      document.getElementById('details').innerText = `
        activation: ${activation}
      `;

    }

    await tf.nextFrame();
  }
}


app();
