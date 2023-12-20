import os
import zipfile
from flask import Flask, render_template, request, session
import matplotlib
matplotlib.use('Agg')
import segment



app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    plot1, plot2 = None, None

    all_folders = session.get('all_folders', [])

    for i in range(1, 4):
        file = request.files.get(f'file{i}')
        if file:
            if file.filename.endswith('.zip'):
                # Create uploads directory if it doesn't exist
                if not os.path.exists(UPLOAD_FOLDER):
                    os.makedirs(UPLOAD_FOLDER)

                # Save the uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Extract the contents of the zip file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(app.config['UPLOAD_FOLDER'], file.filename[:-4]))

                extracted_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename[:-4])

                folders = []
                for fold in os.listdir(extracted_folder_path):
                    folder = os.path.join(extracted_folder_path, fold)
                    folder = folder.replace('\\', '/')
                    folders.append(folder)

                all_folders.append(folders)

            else:
                return "Invalid file format. Please upload a .zip file."

    f1 = request.files['file1'].filename
    f2 = request.files['file2'].filename
    f3 = request.files['file3'].filename
    # print(f1, f2, f3)  # not working!!

    # textbox to enter concentration values
    textbox_value = request.form['textbox']
    text = [str(x.strip()) for x in ''.join(textbox_value).split(',')]

    # pixel intensity plot, heart rate vs concentration plot

    session['all_folders'] = all_folders
    all_folders = session.get('all_folders', [])

    # Creating an instance of class
    s1 = segment.Segmentation(all_folders)

    if request.form['action'] == "Create mean intensity plots":
        # Code to handle intensity plot button click
        plot1 = s1.generate_intensity_plot()

    elif request.form['action'] == "Create heart rate vs concentration plot":
        # plot1 = s1.generate_intensity_plot()

        # Code to handle heart rate vs concentration plot button click
        plot2 = (s1.generate_heartrate_plot(text))

    return render_template('index.html',
                           intensity_plots=plot1,
                           plot_heartrate=plot2,
                           zip1=f1, zip2=f2, zip3=f3,
                           concentrations=textbox_value)


if __name__ == '__main__':
    app.run(debug=True)


