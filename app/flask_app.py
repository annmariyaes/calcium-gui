import os
import zipfile
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import unetsegment



app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/intensity', methods=['GET', 'POST'])
def intensities():
    plot1 = None
    all_folders = []
    f1, f2, f3 = request.files['file1'], request.files['file2'], request.files['file3']
    zip_files = [f1, f2, f3]

    for zip_file in zip_files:
        print(zip_file)
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


    f1.save(secure_filename(f1.filename))
    f2.save(secure_filename(f2.filename))
    f3.save(secure_filename(f3.filename))

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

    # Code to handle button click
    if request.form.get('action') == "Create mean intensity plots":
        plot1 = us1.display_intensity_plot()

    return render_template('index.html',
                           intensity_plots=plot1,
                            zip11=f1.filename,
                            zip21=f2.filename,
                            zip31=f3.filename,
                            chemical1=chemical,
                            fps1=fps,
                            time1=time,
                            times1=time2)




@app.route('/rate', methods=['GET', 'POST'])
def rates():
    plot2 = None
    all_folders = []
    f1, f2, f3 = request.files['file1'], request.files['file2'], request.files['file3']
    zip_files = [f1, f2, f3]

    for zip_file in zip_files:
        print(zip_file)
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


    f1.save(secure_filename(f1.filename))
    f2.save(secure_filename(f2.filename))
    f3.save(secure_filename(f3.filename))

    # textbox to enter chemical
    chemical = request.form['che_textbox']

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
    # s1 = segment.Segmentation(all_folders, chemical, fps, time, times)
    us1 = unetsegment.Unet(all_folders, chemical, fps, time, times)


    # Code to handle button click
    if request.form.get('action') == "Create heart rate vs concentration plot":
        plot2 = us1.display_heartrate_plot(concentrations)



    return render_template('index.html',
                            plot_heartrate=plot2,
                            zip12=f1.filename,
                            zip22=f2.filename,
                            zip32=f3.filename,
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


