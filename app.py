from flask import Flask, request, Response
from LipNet.evaluation.predict import predict

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)


@app.route('/lips-to-text', methods=['POST'])
def lips_to_text():
    video = request.files.get("video")
    if video == None:
        return Response("No video received", status=400)

    vname = "video." + video.filename.split(".")[-1]
    video.save(vname)
    video, result = predict("LipNet/evaluation/models/overlapped-weights368.h5", vname)
    return Response(result)


@app.route("/")
def index():
    return "<h1>LipsReaderAPI...Running<h1>";


if __name__ == "__main__":
    app.run(threaded=False)
