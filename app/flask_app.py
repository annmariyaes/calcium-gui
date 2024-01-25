import os
import zipfile
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
import matplotlib
matplotlib.use('Agg')
import unetsegment


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
all_folders1, all_folders2 = [], []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/intensity', methods=['GET', 'POST'])
def intensities():
    plot1 = None

    zip_files = request.files.getlist('zipfile')

    names = []
    for zip_file in zip_files:
        print(zip_file.filename)
        names.append(secure_filename(zip_file.filename))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(zip_file.filename))
        zip_file.save(file_path)

        with open(file_path, "ab") as f:
            f.seek(int(request.form["dzchunkbyteoffset"]))
            f.write(zip_file.stream.read())

        # Extract the contents of the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename[:-4]))
        extracted_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename[:-4])

        folders = []
        for fold in os.listdir(extracted_folder_path):
            folder = os.path.join(extracted_folder_path, fold)
            folder = folder.replace('\\', '/')
            folders.append(folder)
        all_folders1.append(folders)

    # textbox to enter chemical
    chemical = request.form['chemical1']

    # textbox to enter fps
    fps = request.form['fps1']

    # textbox to enter time duration
    time = request.form['time1_textbox1']

    # textbox to enter range of time duration
    time2 = request.form['time2_textbox1']
    times = [str(x.strip()) for x in ''.join(time2).split('-')]

    # Creating an instance of class
    us1 = unetsegment.Unet(all_folders1, chemical, fps, time, times)

    print(all_folders1)
    print(names)

    # Code to handle button click
    if request.form.get('action') == "Create mean intensity plots":
        plot1 = us1.display_intensity_plot()

    return render_template('index.html',
                           intensity_plots=plot1,
                           chemical1=chemical,
                           fps1=fps,
                           time1=time,
                           times1=time2,
                           names=names)


@app.route('/rate', methods=['GET', 'POST'])
def rates():
    plot2 = None

    zip_files = request.files.getlist('zipfile')

    for zip_file in zip_files:
        print(zip_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(zip_file.filename))
        zip_file.save(file_path)

        with open(file_path, "ab") as f:
            f.seek(int(request.form["dzchunkbyteoffset"]))
            f.write(zip_file.stream.read())

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename[:-4]))
        extracted_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename[:-4])

        folders = []
        for fold in os.listdir(extracted_folder_path):
            folder = os.path.join(extracted_folder_path, fold)
            folder = folder.replace('\\', '/')
            folders.append(folder)
        all_folders2.append(folders)

    # textbox to enter chemical
    chemical = request.form['chemical2']

    # textbox to enter fps
    fps = request.form['fps2']

    # textbox to enter time duration
    time = request.form['time1_textbox2']

    # textbox to enter range of time duration
    time2 = request.form['time2_textbox2']
    times = [str(x.strip()) for x in ''.join(time2).split('-')]

    con_vals = request.form['textbox']
    concentrations = [str(x.strip()) for x in ''.join(con_vals).split(',')]

    # Creating an instance of class
    # s1 = segment.Segmentation(all_folders, chemical, fps, time, times)
    us1 = unetsegment.Unet(all_folders2, chemical, fps, time, times)

    print(all_folders2)
    print(zip_files)

    # Code to handle button click
    if request.form.get('action') == "Create heart rate vs concentration plot":
        plot2 = us1.display_heartrate_plot(concentrations)

    return render_template('index.html',
                            plot_heartrate=plot2,
                            chemical2=chemical,
                            fps2=fps,
                            time2=time,
                            times2=time2,
                            concentrations=con_vals)


'''
from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

@app.route('/download/<upload_id>')
def download(upload_id):
    upload = Upload.query.filter_by(id=upload_id).first()
    return send_file(BytesIO(upload.data),
                     download_name=upload.filename, as_attachment=True)
'''


if __name__ == '__main__':
    app.run(debug=True)


