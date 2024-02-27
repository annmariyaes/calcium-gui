import os
import zipfile
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import unetsegment


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'kist'
all_folders, names  = [], []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/intensity', methods=['GET', 'POST'])
def intensities():
    plot1, plot2 = None, None

    zip_files = request.files.getlist("zipfiles")
    print(request.files)
    print(request.form)

    # Store zip_files in session
    for index, zip_file in enumerate(zip_files):
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
        all_folders.append(folders)

    # textbox to enter chemical
    chemical = request.form['chemical']

    # textbox to enter fps
    fps = request.form['fps']

    # textbox to enter time duration
    time = request.form['time1_textbox']

    # textbox to enter range of time duration
    time2 = request.form['time2_textbox']
    times = [str(x.strip()) for x in ''.join(time2).split('-')]

    con_vals = request.form['textbox']
    concentrations = [str(x.strip()) for x in ''.join(con_vals).split(',')]

    # Creating an instance of class
    us1 = unetsegment.Unet(all_folders, chemical, fps, time, times)
    print(all_folders)

    # Code to handle button click
    if request.form.get('action') == "Create mean intensity plot":
        plot1 = us1.display_intensity_plot()

    # Code to handle button click
    elif request.form.get('action') == "Create heart rate vs concentration plot":
        plot2 = us1.display_heartrate_plot(concentrations)

    return render_template('index.html',
                           chemical=chemical,
                           fps=fps,
                           time=time,
                           times=time2,
                           intensity_plots=plot1,
                           plot_heartrate=plot2,
                           concentrations=con_vals)


if __name__ == '__main__':
    app.run()


