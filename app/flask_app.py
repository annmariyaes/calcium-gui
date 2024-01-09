import os
import zipfile
from flask import Flask, render_template, request, session
import matplotlib
matplotlib.use('Agg')
import unetsegment



app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    plot1, plot2 = None, None

    all_folders = []
    f1, f2, f3 = request.files['file1'], request.files['file2'], request.files['file3']
    zip_files = [f1, f2, f3]

    for zip_file in zip_files:
        if zip_file and zip_file.filename.endswith('.zip'):
                # Create uploads directory if it doesn't exist
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                # Save the uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename)
                zip_file.save(file_path)

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

        else:
            return "Invalid file format. Please upload a .zip file."


    # textbox to enter chemical
    chemical = request.form['che_textbox']

    # textbox to enter fps
    fps = request.form['fps']

    # textbox to enter time duration
    time = request.form['time1_textbox']

    # textbox to enter range of time duration
    time2 = request.form['time2_textbox']
    times = [str(x.strip()) for x in ''.join(time2).split('-')]


    # Creating an instance of class
    # s1 = segment.Segmentation(all_folders, chemical, fps, time, times)
    us1 = unetsegment.Unet(all_folders, chemical, fps, time, times)
    print(all_folders)

    # Code to handle button click
    if request.form['action'] == "Create mean intensity plots":
        plot1 = us1.display_intensity_plot()

    elif request.form['action'] == "Create heart rate vs concentration plot":
        plot1 = us1.display_heartrate_plot()


    return render_template('index.html',
                           intensity_plots=plot1,
                           zip1=f1.filename,
                           zip2=f2.filename,
                           zip3=f3.filename,
                           chemical=chemical,
                           fps=fps,
                           time=time,
                           times=time2)


if __name__ == '__main__':
    app.run(debug=True)


