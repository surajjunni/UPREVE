from flask import redirect,url_for,Flask,request,make_response,render_template,jsonify
import json
import sqlite3
import glob
import os
import csv
import openpyxl
from werkzeug.utils import secure_filename
import subprocess
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import datetime
from zipfile import ZipFile
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import quote,unquote
import zipfile
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from networkx.readwrite import json_graph
import shutil
import io
plt.switch_backend('agg')

app=Flask(__name__)

prints=[]


def generate_heatmap(data, timestamp, filename, config, dataset_name):
    # Generate heatmap
    plt.clf()
    result_matrix=data
    data = data.dropna()
    correlation_matrix = data.corr()
    correlation_matrix.interpolate(method='linear', axis=1, inplace=True)

    # Fill the remaining NaN values with zeros
    correlation_matrix.fillna(0, inplace=True)
    fig = px.imshow(correlation_matrix, color_continuous_scale='Viridis',text_auto=True)
    fig.update_layout(
        width=1000,
        height=1000,
        xaxis_tickangle=-45,
        title=f"{filename} Heatmap",
        title_x=0.5,
        font=dict(size=18)
    )

    fig_name = f"{filename}_heatmap.png"
    fig_dir = f"static/images/{timestamp}"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = fig_dir + "/" +fig_name
    pio.write_image(fig, fig_path)
    variables = data.columns
    corr_matrix = data.corr()
    outliers = []
    for col in corr_matrix.columns:
        q1 = corr_matrix[col].quantile(0.25)
        q3 = corr_matrix[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers.extend(corr_matrix[(corr_matrix[col] < lower_bound) | (corr_matrix[col] > upper_bound)].index)

    # Remove duplicates from the list of outliers
    outliers = list(set(outliers))

    

    # Get the list of variable names
    variable_names = data.columns
    least_correlated_dict = {}


    for variable in variable_names:
        min_correlation = np.inf
        least_correlated_variable = None
        max_correlation = -np.inf
        strongest_correlated_variable = None

        # Find the least and strongest correlated variables for the current variable
        for other_variable in variable_names:
            if variable != other_variable:
                correlation = correlation_matrix.loc[variable, other_variable]
                if correlation < min_correlation:
                    min_correlation = correlation
                    least_correlated_variable = other_variable
                if correlation > max_correlation:
                    max_correlation = correlation
                    strongest_correlated_variable = other_variable

        # Store the least and strongest correlated variables in the dictionary
        least_correlated_dict[variable] = {
            "least_correlated_variable": least_correlated_variable,
            "strongest_correlated_variable": strongest_correlated_variable,
        }

    # # Get the connected components of the graph
    # connected_components = list(nx.strongly_connected_components(G))
    # component_map = {}
    # # Print the connected components in the meta-data format
    # for idx, nodes in enumerate(connected_components):
    #     node_list = []
    #     for node in nodes:
    #         node_list.append(node)
    #     component_map[idx] = node_list
    # # Get the strength of the causal relationships
    # strengths = data.values


    # # Convert int64 to regular Python integers
    # strengths = strengths.astype(int).tolist()

    # Generate meta information
    meta_info = {
        "Column names": variables.tolist(),  # Convert Index to list
        "outlier columns" : outliers,
        "strongest and weakest variables": least_correlated_dict, 
    }

    # Serialize the data to JSON
    json_data = json.dumps(meta_info)

    
    meta_dir = f"static/images/{timestamp}"
    os.makedirs(meta_dir, exist_ok=True)
    # Save metadata to file
    meta_data_path = os.path.join(meta_dir, f"{filename}_heatmap_meta_data.json")
    #os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_data_path, 'w') as f:
        json.dump(meta_info, f)

    
    # Save config
    config_path = os.path.join(meta_dir, f"{filename}_config.json")
    if(not os.path.exists(config_path)):
        with open(config_path, 'w') as f:
            json.dump(config, f)

    # Save dataset name
    dataset_name_path = os.path.join(meta_dir, f"{filename}_dataset_name.txt")
    if(not os.path.exists(dataset_name_path)):
        with open(dataset_name_path, 'w') as f:
            f.write(dataset_name)    

    plt.clf()
    return meta_info, quote(fig_path)


def generate_directed_graph(data, timestamp, filename, config, dataset_name):
    # Generate directed graph
    plt.clf()
    G = nx.DiGraph()
    for col in data.columns:
        G.add_node(col)

    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i != j and col1 != 'Class' and col2 != 'Class':
                corr = data[[col1, col2]].corr().iloc[0, 1]
                if corr >= 0.5:
                    G.add_edge(col1, col2, weight=corr)

    pos = nx.spring_layout(G, k=2.15, iterations=20)
    node_sizes = [2500 * len(col) for col in G.nodes]
    node_colors = ['#1f78b4' if col != 'Class' else '#33a02c' for col in G.nodes]
    edge_colors = ['black' if w > 0 else '#fb8072' for u, v, w in G.edges(data='weight')]
    edge_widths = [25 * w ** 2 for u, v, w in G.edges(data='weight')]
    plt.figure(figsize=(30, 30))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,arrowstyle='->', arrowsize=25)
    nx.draw_networkx_labels(G, pos, font_size=22, font_family='sans-serif', font_color='black')

    # Save image and other files
    fig_name = f"{filename}_graph.png"
    fig_dir = f"static/images/{timestamp}"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = fig_dir + "/" +fig_name
    plt.savefig(fig_path)
    plt.clf()
    variables = data.columns
    G = nx.DiGraph(np.array(data.values))
    node_names = {i: col for i, col in enumerate(data.columns)}
    G = nx.relabel_nodes(G, node_names)
    edge_number=len(G.edges)
    density=nx.density(G)
    density=round(density,2)
    causes = nx.to_dict_of_lists(G)
    causes = {k: v for k, v in causes.items() if len(v)!=0}
    #graph_json=json_graph.dumps(DG)
    # # Get the connected components of the graph
    # connected_components = list(nx.strongly_connected_components(G))
    # component_map = {}
    # # Print the connected components in the meta-data format
    # for idx, nodes in enumerate(connected_components):
    #     node_list = []
    #     for node in nodes:
    #         node_list.append(node)
    #     component_map[idx] = node_list
    # # Get the strength of the causal relationships
    # strengths = data.values


    # # Convert int64 to regular Python integers
    # strengths = strengths.astype(int).tolist()

    # Generate meta information
    meta_info = {
        "Node labels": variables.tolist(),  # Convert Index to list
        "Number of edges":edge_number,
        "Desnsity of the directed graph":density,
        "causes list":causes,
    }


    meta_dir = f"static/images/{timestamp}"
    os.makedirs(meta_dir, exist_ok=True)

    # Save meta data
    meta_data_path = os.path.join(meta_dir, f"{filename}_graph_meta_data.json")
    with open(meta_data_path, 'w') as f:
        json.dump(meta_info, f)

   # Save config
    config_path = os.path.join(meta_dir, f"{filename}_config.json")
    if(not os.path.exists(config_path)):
        with open(config_path, 'w') as f:
            json.dump(config, f)

    # Save dataset name
    dataset_name_path = os.path.join(meta_dir, f"{filename}_dataset_name.txt")
    if(not os.path.exists(dataset_name_path)):
        with open(dataset_name_path, 'w') as f:
            f.write(dataset_name)

    print(fig_path)
    return meta_info, quote(fig_path)



def get_column_type(column_name, dataset_name):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute(f"SELECT column_type FROM column_types WHERE name='{dataset_name}' AND column_name='{column_name}'")
    column_type = cur.fetchone()[0]
    conn.close()
    return column_type

def convert_ordinal(df,column, mapping_order):
    mapping = dict(zip(mapping_order[column].str.split(':', expand=True)[1], mapping_order[column].str.split(':', expand=True)[0].astype(int)))

    print(mapping)
    print(df[column])
    df[column] = df[column].apply(lambda x: mapping[x])
    return df

@app.route('/algorithm/config/<algorithm>', methods=['GET'])
def get_algorithm_config(algorithm):
    config_path = os.path.join('script', 'algorithms', algorithm, 'config.json')
    print(config_path)
    df=pd.read_json(config_path)
    print(df)
    if not os.path.exists(config_path):
        return jsonify({'error': 'Algorithm config not found'}), 404
    
    with open(config_path) as file:
        config = json.load(file)

    config['parameters'] = [param for param in config['parameters'] if param['name'] not in ['input', 'output']]   
    
    return jsonify(config)

@app.route('/',methods=["POST","GET"])
def index():
    print("here")
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # query problem_type table for available problem types
    try:
        dataset = c.execute('''SELECT name FROM column_types''').fetchall()
    except:
        dataset=[[None]]    
    algo_name = c.execute("SELECT name  FROM  scripts WHERE belong_to = ?", ("algorithm",)).fetchall()
    #problem = c.execute('''SELECT name  FROM  scripts where belong_to=?''').fetchall()
    #metrics=c.execute("SELECT name  FROM  scripts WHERE belong_to = ?", ("metric",)).fetchall()
    dataset = [i[0] for i in dataset]
    dataset = sorted(list(set(dataset)))

    algo_name_list  = [i[0] for i in algo_name ]
    print(algo_name_list)
    #problem = sorted(list(set(problem))) + ["None"]
    #metrics=[i[0] for i in metrics]
    #metrics=list(set(metrics))
    #print(dataset,problem)
    #print(algorithms,problem_ty
    return render_template("index.html",ds=dataset,alg_name=algo_name_list)

@app.route('/algorithm',methods=["POST","GET"])
def algo_support():
    algorithm_folders = os.listdir('script/algorithms')
    algorithms = []
    for folder in algorithm_folders:
        if os.path.isfile(f'script/algorithms/{folder}/config.json'):
            with open(f'script/algorithms/{folder}/config.json') as f:
                config = json.load(f)
            # Remove "input" and "output" parameters from the list
            parameters = [p for p in config['parameters'] if p['name'] not in ['input', 'output']]
            desc=None
            if(config['description']):
                desc=config['description']
            #print(desc)    
            algorithms.append({
                'name': config['algorithm_name'],
                'description': desc,
                'parameters': parameters,
                'script_lang': config['script_lang']
            })
            print(algorithms)
    return render_template('algorithm.html', algorithms=algorithms) 

@app.route('/metric',methods=["POST","GET"])
def metric_support():
    algorithm_folders = os.listdir('script/metrics')
    algorithms = []
    for folder in algorithm_folders:
        if os.path.isfile(f'script/metrics/{folder}/config.json'):
            with open(f'script/metrics/{folder}/config.json') as f:
                config = json.load(f)
            # Remove "input" and "output" parameters from the list
            desc=None
            try:
                desc=config['description']
            except:    
                pass
            parameters = [p for p in config['parameters'] ]
            algorithms.append({
                'name': config['algorithm_name'],
                'description':desc,
                'parameters': parameters,
                'script_lang': config['script_lang']
            })       
    return render_template('metrics.html', algorithms=algorithms)

@app.route('/dataset',methods=["POST","GET"])
def dataset_support():
    dataset_info = []
    data_directory = 'static'

    # Iterate through each file in the data directory
    directory=set(os.listdir(data_directory))
    print(directory)
    for filename in directory:
        if filename.endswith('.csv') and filename.endswith('_processed.csv'):
            file_path = os.path.join(data_directory, filename)
            file_info = {}

            # Read the data from the CSV file using pandas
            df = pd.read_csv(file_path)

            # Get the shape (dimensions) of the data
            data_shape = df.shape

            # Read the column names from the CSV file
            column_names = df.columns.tolist()
            dataset_name = os.path.splitext(filename)[0].split('_')[0]

            # Prepare the dataset information
            file_info['filename'] = dataset_name
            file_info['shape'] = data_shape
            file_info['column_names'] = column_names
            ground_truth_file = f"{dataset_name}_groundtruth.csv"
            order_file = f"{dataset_name}order.csv"
            print(ground_truth_file,order_file)
            # Set boolean values based on file existence
            dir=os.path.join(data_directory, ground_truth_file)
            dir1=os.path.join(data_directory, order_file)
            val=os.path.exists(dir)
            val1=os.path.exists(dir1)
            print(val,val1)
            file_info['has_ground_truth'] = val
            file_info['has_order'] = val1
            # Add the dataset information to the list
            dataset_info.append(file_info)

    return render_template('dataset.html', dataset_info=dataset_info)           

@app.route('/script_entry',methods=["POST","GET"])
def script_entry():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    problem = c.execute('''SELECT problem_type FROM problem_table''').fetchall()
    problem=[i[0] for i in problem]
    problem=list(set(problem))+["None"]
    return render_template("script.html",problem_type=problem)

@app.route('/data_entry',methods=["POST","GET"])
def data_entry():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    problem = c.execute('''SELECT problem FROM problem_table''').fetchall()
    problem = [i[0] for i in problem]
    problem = sorted(list(set(problem))) + ["None"]
    return render_template("dataset_upload.html",problem=problem)    

@app.route('/problem_entry',methods=["POST","GET"])
def problem_entry():
    return render_template("problem.html",message=None)

@app.route('/metric_entry',methods=["POST","GET"])
def metric_entry():
    return redirect(url_for('upload_metrics'))  

@app.route('/visualize/static/<path:filename>',methods=["POST","GET"])
def visual_entry(filename):
    return render_template("visualization.html",fig_paths=None,matrix=None,visualization_type=None,config=None,dataset=None,meta_info=None,extras=quote("static/"+filename),result=None,settings=None)    

@app.route('/metric/static/<path:filename>',methods=["POST","GET"])
def metrics_entry(filename):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # query problem_type table for available problem types
    metrics=c.execute("SELECT name  FROM  scripts WHERE belong_to = ?", ("metric",)).fetchall()
    metrics=[i[0] for i in metrics]
    metrics=list(set(metrics))
    #metrics = [s.replace("_", " ") for s in metrics]
    dic={}
    for i in metrics:
        dic[i]="unchecked"
    #print(dataset,problem)
    return render_template("metric_evaluate.html",metric=dic,extras=quote("static/"+filename),config=None,dataset=None,result=None,csv_files=None)

@app.route('/prints')
def get_prints():
    return jsonify({'prints': prints})    

@app.route('/upload_metrics', methods=['GET', 'POST'])
def upload_metrics():
    if request.method == 'POST':
        #algorithm = request.form['algorithm']
        problem_type = request.form['problem_type']
        metric_file = request.form['metric_file']
        #metric_data = metric_file.read()

        conn = sqlite3.connect('database.db')
        c = conn.cursor()

        # insert metric data into the metrics table
        c.execute('''INSERT INTO metrics (metric_name, problem_type)
                     VALUES (?, ?)''', (metric_file, problem_type))
        conn.commit()

        return 'Metric uploaded for algorithm "{}" and problem type "{}"'.format(metric_file, problem_type)

    else:
        print("here")
        conn = sqlite3.connect('database.db')
        c = conn.cursor()

        # query problem_type table for available problem types
        problem_types = c.execute('''SELECT problem FROM problem_table''').fetchall()
        c.execute('''SELECT name FROM scripts WHERE belong_to = ?''', ('metric',))
        metric_names = c.fetchall()
        #print(algorithms,problem_types)
        return render_template('metric.html',problem_types=problem_types,metric_names=metric_names)

@app.route('/upload_problem_type', methods=['POST'])
def upload_problem():
    problem_type = request.form['problem_type']
    problem = request.form['problem']
    mesg=None
    # Store the problem type in the database
    try:
        with sqlite3.connect('database.db') as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO problem_table (problem,problem_type) VALUES (?,?)", (problem,problem_type))
            conn.commit()
            mesg="success"
    except Exception as e:
        mesg="error"
    print(mesg)
    return render_template("problem.html",message=mesg)    

@app.route('/save_script', methods=['POST'])
def save_script():
    # get the uploaded zip file
    if request.method == 'POST':
        try:
            script_name = request.form.get('script-name')
            script_type = request.form.get('script-type')
            uploaded_file = request.files['script-file']

            # save the file to disk
            if script_type == "algorithm":
                filep="script/algorithms/"
                file_path = filep + secure_filename(uploaded_file.filename)
            else:
                filep="script/metrics/"
                file_path = filep + secure_filename(uploaded_file.filename)

            uploaded_file.save(file_path)
            zip = ZipFile(file_path)
            zip.extractall(filep)
            path1=f"{filep}{uploaded_file.filename[:-4]}"
            print(os.listdir(path1))
            with open(f'{path1}/config.json', 'r') as f:
                config = json.load(f)
            # create a virtual environment for the project
            env_path = os.path.join('envs', config['algorithm_name'])
            if not os.path.exists(env_path):
                subprocess.call(['python', '-m', 'venv', env_path])
            print("env created")
            # activate the virtual environment
            activate_script=os.path.join(env_path,'bin')
            print(activate_script)
            # subprocess.call(activate_script,shell=True)
            print("called")
            # install packages for the specified language
            if config['script_lang'] == 'Python':
                # install Python packages
                requirements_path = f"{path1}/{config['requirements_file']}"
                print(requirements_path)
                subprocess.call(f'{activate_script}/pip install -r {requirements_path}',shell=True)

            elif config['script_lang'] == 'R':
                # install R packages
                packages = []
                with open(f"{path1}/{config['requirements_file']}") as f:
                    packages = [line.strip() for line in f if line.strip() != ""]
                print(packages)    
                for package in packages:
                    subprocess.call(['R', '-e', f'install.packages("{package}",repos="https://cran.r-project.org")'])
            # subprocess.run(["deactivate"], shell=True)
            print("database")
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            # create the scripts table if it doesn't exist
            c.execute('''CREATE TABLE IF NOT EXISTS scripts (
                        name TEXT PRIMARY KEY,belong_to TEXT,algo_path TEXT)''')
            
            # insert the script name into the database
            result = c.execute("INSERT INTO scripts VALUES (?,?,?)",(script_name,script_type,path1))
            conn.commit()
            # close the database connection
            conn.close()
                    
            return 'Script updated'

        except Exception as e:
             error_message = str(e)
             print(error_message)
             return error_message, 400

def get_hyperparams(algo_name):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM hyperparams WHERE name = ?", (algo_name,))
    result = cursor.fetchone()
    conn.close()
    return result

def validate_params(params, hyperparams):
    if len(params) > len(hyperparams):
        return False
    for i in range(len(params)):
        for key in params[i]:
            if key not in hyperparams[i]:
                return False
    return True

@app.route('/get_datasets/<setting>', methods=['GET'])
def get_datasets(setting):
    data_folders = os.listdir(os.path.join('results', setting))
    return jsonify(data_folders)

@app.route('/causal-matrix/static/<path:filename>', methods=["POST", "GET"])
def causal_matrix(filename):
    file = "static/" + filename
    data = pd.read_csv(file)
    mapping = {i: col for i, col in enumerate(data.columns)}
    output = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                edge = {'source': mapping[i], 'target': mapping[j], 'value': int(data.iloc[i][j])}
                output.append(edge)
    return jsonify(output)    

@app.route("/get_timestamps/<setting>/<dataset>",methods=['GET'])
def get_timestamps(setting, dataset):
    timestamp_dir = os.path.join("results", setting, dataset)
    timestamps = [name for name in os.listdir(timestamp_dir) if os.path.isdir(os.path.join(timestamp_dir, name))]
    return jsonify(timestamps)

@app.route('/metrics_evaluate', methods=['POST'])
def metrics_evaluate():
    result_path=unquote(request.form['file_path'])
    print(result_path)
    result_path1=result_path
    uploaded_file= request.form
    print("checkbox",uploaded_file)
    metric_list=[]
    dic={}
    for k,v in uploaded_file.items():
        print(k)
        if(v=="on"):
            metric_list.append(k)
    print(metric_list)
    #metric_list = [s.replace(" ", "_") for s in metric_list]
    # Connect to the database and fetch the config, settings, dataset, and timestamp for the result_path
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT config,settings,dataset FROM json_data WHERE result_path=?", (result_path,))
    result = c.fetchone()
    print(result)
    config, settings, dataset = result
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
    # Extract the input CSV files from the result.zip file
    input_files = []
    # Create the output directory if it doesn't exist
    out_dir = f"static/metrics/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    result_path=result_path.split('/')[1]
    result_path_withoutzip='results/'+ result_path.split('_')[0]
    print(result_path_withoutzip)
    for filename in os.listdir(result_path_withoutzip):
        if filename.endswith('.csv'):
            #src_path = os.path.join(result_path_withoutzip, filename)
            #dst_path = os.path.join(out_dir, filename)
            #shutil.copy(src_path, dst_path)
            input_files.append(f'{result_path_withoutzip}/{filename}')

    input_file = "--input " + ' '.join([f'"{filename}"' for filename in input_files])
    ground_truth = f'--ground_truth  static/{dataset}_groundtruth.csv' 

    
    print("run metric")
    # Loop through the selected metrics and run the corresponding script
    for metric_name in metric_list:
        c.execute("SELECT algo_path FROM scripts WHERE name = ?", (metric_name,))
        algo_path = c.fetchone()[0]
        with open(os.path.join(algo_path, "config.json"), 'r') as f:
            row_data = json.load(f)
        
        # Activate the virtual environment and run the script
        env_path = 'envs/'+row_data['algorithm_name']
        activate_script=os.path.join(env_path,'bin')
        cmd = f'{activate_script}/python {algo_path}/{row_data["script_path"]}'
        cmd+=f' {input_file}' 
        print(input_file)
        cmd+=f' {ground_truth} --output "{out_dir}/{metric_name}.csv"'
        print(cmd)
        try:
            subprocess.run(cmd,shell=True)
        except:
            continue
    
     # Save config
    # query problem_type table for available problem types
    csv_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
    for filename in os.listdir(result_path_withoutzip):
        if filename.endswith('.csv'):
            src_path = os.path.join(result_path_withoutzip, filename)
            dst_path = os.path.join(out_dir, filename)
            shutil.copy(src_path, dst_path)
            #input_files.append(f'{result_path_withoutzip}/{filename}')
    metrics=c.execute("SELECT name  FROM  scripts WHERE belong_to = ?", ("metric",)).fetchall()
    metrics=[i[0] for i in metrics]
    metrics=list(set(metrics))
    #metrics = [s.replace("_", " ") for s in metrics]
    print(metrics)
    for i in metrics:
        if(i in metric_list):
            dic[i]="checked"
        else:
            dic[i]="unchecked"
            
    config_path = os.path.join(out_dir, f"config.json")
    if(not os.path.exists(config_path)):
        with open(config_path, 'w') as f:
            json.dump(config, f)


    # Save dataset name
    dataset_name_path = os.path.join(out_dir, f"dataset_name.txt")
    if(not os.path.exists(dataset_name_path)):
        with open(dataset_name_path, 'w') as f:
            f.write(dataset)
            f.write(settings)                    
    c.close()
    zip_path=f"static/metrics/{timestamp}_metrics.zip"
    zipf = ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(out_dir):
        for file in files:
               zipf.write(os.path.join(root, file))
    zipf.close()
    
    data = []

    # Read data from each CSV file
    for file in csv_files:
        with open(f'{out_dir}/{file}', 'r') as f:
            csv_reader = csv.reader(f)
            csv_data = [row for row in csv_reader]
            data.append((file[:-4], csv_data))

    config = json.loads(config)        
    # config=json.dumps(config, indent=4)
    return render_template('metric_evaluate.html',metric=dic,config=config,dataset=dataset,extras=quote(result_path1),result=quote(zip_path),csv_files=data) 


@app.route('/visualization', methods=['POST'])
def generate_visualization():
    file_path=unquote(request.form['file_path'])
    visualization_type = request.form['visualization']
    print(file_path)
    print(visualization_type)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")
    fig_paths = []
    meta_info =[]
    timestamp=file_path.split('/')[1].split('_')[0]
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT config,settings,dataset FROM json_data WHERE result_path=?", (file_path,))
    result = c.fetchone()  # get the first row of the result
    c.close()

    config,settings,dataset = result
    config = json.loads(config)

    
    if(settings=="normal"):
        i=0
        with zipfile.ZipFile(file_path, 'r') as myzip:
             for filename in myzip.namelist():
                 if filename.endswith('.csv'):
                    print(filename)
                    filen = os.path.basename(filename)
                    filen_without_ext = os.path.splitext(filen)[0]
                    with myzip.open(filename) as myfile:
                         fig_path=None
                         data = pd.read_csv(myfile)
                         print(data)
                         if visualization_type == 'heatmap':
                            meta,fig_path = generate_heatmap(data, timestamp,filen_without_ext,config[i],dataset)
                         elif visualization_type == 'directed-graph':
                            print("directed_graph")
                            meta,fig_path = generate_directed_graph(data, timestamp,filen_without_ext,config[i],dataset)
                         if(fig_path): 
                            if(visualization_type=="directed-graph"):
                                f1=quote(os.path.join(f"static/images/{timestamp}",filen))
                            else:
                                f1=fig_path.split() 
                            fig_paths.append(f1)
                            meta_info.append(meta)
                    i+=1

    else:

        i=0
        with zipfile.ZipFile(file_path, 'r') as myzip:
             for filename in myzip.namelist():
                 if filename.endswith('.csv'):
                    print(filename)
                    filen = os.path.basename(filename)
                    filen_without_ext = os.path.splitext(filen)[0]
                    with myzip.open(filename) as myfile:
                         fig_path=None
                         data = pd.read_csv(myfile)
                         print(data)
                         if visualization_type == 'heatmap':
                            meta,fig_path = generate_heatmap(data, timestamp,filen_without_ext,config,dataset)
                         elif visualization_type == 'directed-graph':
                            print("directed_graph")
                            meta,fig_path = generate_directed_graph(data, timestamp,filen_without_ext,config,dataset)
                         if(fig_path):  
                            if(visualization_type=="directed-graph"):
                               f1=quote(os.path.join(f"static/images/{timestamp}",filen))
                            else:
                               f1=fig_path.split()     
                            fig_paths.append(f1)
                            meta_info.append(meta)
                    i+=1

    
        
    print(fig_paths)
    result_path=file_path.split('/')[1]
    result_path_withoutzip='results/'+ result_path.split('_')[0]
    print(result_path_withoutzip)
    for filename in os.listdir(result_path_withoutzip):
        if filename.endswith('.csv'):
            src_path = os.path.join(result_path_withoutzip, filename)
            dst_path = os.path.join(f"static/images/{timestamp}", filename)
            shutil.copy(src_path, dst_path)
    zip_path=f"static/images/{timestamp}_visualize.zip"
    zipf = ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(f"static/images/{timestamp}"):
        for file in files:
            if root == f"static/images/{timestamp}":
               zipf.write(os.path.join(root, file))
    zipf.close()
    print(fig_paths)
    print(config)
    return render_template('visualization.html', fig_paths=fig_paths,visualization_type=visualization_type,config=config,dataset=dataset,meta_info=meta_info,extras=quote(file_path),result=quote(zip_path),settings=settings)           

@app.route('/upload_json', methods=['POST'])
def upload_json():
    data=request.form
    config = []

    # Extract algorithm names from the 'algorithm' field
    algorithm_names = data.get('algorithm').split(',')

    for algorithm_name in algorithm_names:
        parameters = []

        # Extract parameter values for the current algorithm
        for key, value in data.items():
            if key.startswith('parameters[' + algorithm_name + ']'):
                parameter_name = key.split('[' + algorithm_name + '][')[1].split(']')[0]
                
                # Convert integer and float values to numbers
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif value == 'None':
                   value = None
                elif value == 'true':
                    value = True
                elif value == 'false':
                    value = False
                else:
                    try:
                        value=int(value)
                    except:
                        pass    
                parameters.append({'name': parameter_name, 'value': [value]})

        # Add the algorithm and its parameters to the config list
        config.append({'algorithm_name': algorithm_name, 'parameters': parameters})

    # Convert the config list to JSON
    json_config = json.dumps(config, indent=2)

    # Print the JSON config
    print(data)
    print(json_config)
    log=[]
    #print("file here",file)
    # Load the JSON data
     # Load the JSON data
    try:
        data = json.loads(json_config)
    except:
        log.append("Unsupported type file.Please upload the correct type.")
        return jsonify({'error': log })
    file_json=data
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    now = datetime.datetime.now()
    # Format the date and time to use as the folder name
    log.append(f'Dataset : {request.form["datasets"]}')
    folder_name = now.strftime("%Y-%m-%d %H-%M-%S")
    
    for algo in data:
        print(algo)
        try:
            algo_name = algo["algorithm_name"]
        except:
            log.append("Please upload the correct json.Use the sample json as refernce")
            return jsonify({'error': log })

        log.append(f"Algorithm : {algo_name}")
        # Check if the algorithm is present in the hyperparams table
        c.execute("SELECT algo_path FROM scripts WHERE name = ?", (algo_name,))
        row = c.fetchone()
        if row is None:
            log+=f"Algorithm {algo_name} not found in database"
            return jsonify({'error': log})
        
        algo_path=row[0]
        row=(row[0]+"/config.json")
        print(row)
        with open(row, 'r') as f1:
             row_data = json.load(f1)
        print(row_data)
        try:
            params=algo["parameters"]
        except:
            log.append("Please upload the correct json.Use the sample json as refernce")
            return jsonify({'error': log })

        log.append("Checking if all parameters in the config file are present in the required_params column")
        
        print(params,row_data)
        print("Settings: normal")
        # if(request.form["optradio"]=="normal"):
        input_file=f' --input static/{request.form["datasets"]}_processed.csv'
        dataset_name=request.form["datasets"]
        alg_name=row_data["algorithm_name"]
        out_dir=f"results/"
        output_dir = out_dir+"/" + folder_name
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir)
        env_path = 'envs/'+row_data['algorithm_name']
        # activate the virtual environment
        log.append("activating the virtual environment")
        activate_script=os.path.join(env_path,'bin')
        print(activate_script)
        # subprocess.call([activate_script],shell=True)
        import sys
        print(sys.prefix)   
        if(row_data["script_lang"]=="Python"):
                log.append(f"running the script")
                cmd = f'{activate_script}/python {algo_path}/{row_data["script_path"]}'
                print(cmd)
                cmd+=input_file
                log.append(f"Parameters value")
                for param in params:
                    log.append(f"{param['name']} = {param['value'][0]}")
                    cmd+=f" --{param['name']} {param['value'][0]}"
                cmd+=f' --output "{output_dir}/{algo_name}.csv"'
                print(cmd)
                try:
                    result=subprocess.run(cmd,shell=True,capture_output=True,text=True)
                except:
                    log.append(f'Input/Output parameters is missing.Please upload a new script')
                    return jsonify({'error': log})
                if(result.returncode!=0):
                   log.append(result.stderr)
                   return jsonify({'error': log})   
        else:
            cmd= f'Rscript {algo_path}/{row_data["script_path"]}'
            cmd+=input_file
            log.append(f"Parameters value")
            for param in params:
                log.append(f"{param['name']} = {param['value'][0]}")
                cmd+=f" --{param['name']} {param['value'][0]}"
            cmd+=f' --output "{output_dir}/{algo_name}.csv"'
            print(cmd)
            try:
                result=subprocess.run(cmd,shell=True,capture_output=True,text=True)
            except:
                log.append(f'Input/Output parameters is missing.Please upload a new script')
                return jsonify({'error': log})
                #cmd+=f' --output "{output_dir}/{algo_name}.csv"'

            if(result.returncode!=0):
                log.append(result.stderr)
                return jsonify({'error': log})  
        log.append(f"Ran successfully.Output is saved as {algo_name}.csv")     

        # else:
        #     print("hyperparameter setting")
        #     input_file=f' --input static/{request.form["datasets"]}_processed.csv'
        #     dataset_name=request.form["datasets"]
        #     alg_name=row_data["algorithm_name"]
        #     out_dir=f"results/"
        #     now = datetime.datetime.now()
        #     # Format the date and time to use as the folder name
        #     folder_name = now.strftime("%Y-%m-%d %H-%M-%S")
        #     output_dir = out_dir+"/" + folder_name
        #     if(not os.path.exists(output_dir)):
        #        os.makedirs(output_dir)
        #     env_path = 'envs/'+row_data['algorithm_name']
        #     # activate the virtual environment
        #     log.append("activating the virtual environment")
        #     activate_script=os.path.join(env_path,'bin')
        #     print(activate_script)
        #     # subprocess.call([activate_script],shell=True)
        #     import sys
        #     print(sys.prefix) 
        #     l=len(params[0]["value"])
        #     log.append(f"Parameters value")
        #     for i in range(l):  
        #         if(row_data["script_lang"]=="Python"):
        #              log.append(f"running the script for param {i}")
        #              cmd = f'{activate_script}/python {algo_path}/{row_data["script_path"]}'
        #              print(cmd)
        #              cmd+=input_file
        #              for param in params:
        #                  log.append(f"{param['name']} = {param['value'][i]}")
        #                  cmd+=f" --{param['name']} {param['value'][i]}"
        #              cmd+=f' --output "{output_dir}/{algo_name}_{i}.csv"'
        #              print(cmd)
        #              try:
        #                 result=subprocess.run(cmd,shell=True,capture_output=True,text=True)
        #              except:
        #                 log.append(f'Input/Output parameters is missing.Please upload a new script')
        #                 return jsonify({'error': log})
        #              print(result)   
        #              if(result.returncode!=0):
        #                 log.append(result.stderr)
        #                 return jsonify({'error': log})   
        #         else:
        #             log.append(f"running the script for param {i}")
        #             cmd= f'Rscript {algo_path}/{row_data["script_path"]}'
        #             cmd+=input_file
        #             for param in params:
        #                 log.append(f"{param['name']} = {param['value'][i]}")
        #                 cmd+=f" --{param['name']} {param['value'][i]}"
        #             cmd+=f' --output "{output_dir}/{algo_name}_{i}.csv"'
        #             print(cmd)
        #             try:
        #                 result=subprocess.run(cmd,shell=True,capture_output=True,text=True)
        #             except:
        #                 log.append(f'Input/Output parameters is missing.Please upload a new script')
        #                 return jsonify({'error': log})
        #             if(result.returncode!=0):
        #                 log.append(result.stderr)
        #                 return jsonify({'error': log})     



             
        #         log.append(f"Ran successfully for param {i}")
    

     # Save config
    config_path = os.path.join(output_dir, f"config.json")
    if(not os.path.exists(config_path)):
        with open(config_path, 'w') as f:
            json.dump(file_json, f)

    # Save dataset name
    dataset_name_path = os.path.join(output_dir, f"dataset_name.txt")
    if(not os.path.exists(dataset_name_path)):
        with open(dataset_name_path, 'w') as f:
            f.write(dataset_name)
    zipf = ZipFile(f"static/{folder_name}_results.zip", 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()
    with open(f'static/{folder_name}_results.zip', 'rb') as f:
               d1 = f.read()                      
    
    print("json_data")
    print(data)
    # file=request.files['file']
    # d=json.loads(file.read())
    file_json=json.dumps(file_json)                                         
    c.execute('''CREATE TABLE IF NOT EXISTS json_data 
             (config TEXT, settings text, dataset text, timestamp text, result blob,result_path text)''')
    print(file_json, dataset_name, folder_name,sqlite3.Binary(d1))
    res=c.execute('''INSERT INTO json_data 
                 (config, settings,dataset, timestamp,result,result_path) 
                 VALUES (?,?,?,?,?,?)''', (file_json,"normal",dataset_name, folder_name,sqlite3.Binary(d1),f'static/{folder_name}_results.zip')) 
    conn.commit()
    conn.close()
    print(res)
    flag="False"
    if(os.path.exists(f'static/{dataset_name}_groundtruth.csv')):
        flag="True"
    print(flag)    
    return jsonify({'success': True, 'output': log ,'result':f"static/{folder_name}_results.zip",'flag':flag})   

@app.route('/upload_dataset', methods=["POST"])
def upload_dataset():
    print(request.form)
    file = request.files['dataset']
    order= request.files['column_order']
    name = request.form['name']
    type = request.form['type']
    problems = request.form.getlist('problem_name')
    #ground_truth=request.form['ground_truth']
    #conn.close()
    df=pd.read_csv(io.String(file.read().decode('utf-8')))    
    # Determine the data types for each column
    types = {}
    for col in df.columns:
        col_parts = col.split(':')
        if len(col_parts) == 2:
            col_name, col_type = col_parts
            types[col] = col_type.strip()
            
    
    # remove the first row from the dataframe
    print(types.keys())
    # calculate the range of each column based on its type
    ranges = {}
    for col in types.keys():
        if types[col] == 'numeric':
            print(ranges,col)
            ranges[col]=(df[col].min(), df[col].max())
        else:
            ranges[col] = None
    # Open the file using the appropriate library
    if file.filename.endswith('.csv'):
        # Save the file to a temporary location
        file_path = os.path.join("static", name+".csv")
        file.save(file_path)
        df=pd.read_csv(file_path)
            
    elif file.filename.endswith('.xlsx'):
        # Save the file to a temporary location
        file_path = os.path.join("static", name+".xlsx")
        file.save(file_path)
        df=pd.read_csv(file_path)
        
    else:
        return 'Invalid file type'

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute(f'CREATE TABLE IF NOT EXISTS `data_metric` (name TEXT, problem TEXT)')
    for problem in problems:
        print(problem)
        c.execute(f'INSERT INTO data_metric VALUES (?, ?)', (name, problem))
    conn.commit()
    # Save the data to the database
    c.execute(f'CREATE TABLE IF NOT EXISTS `column_types` (name TEXT, column_name TEXT, column_type TEXT, ranges TEXT)')
    for k,v in types.items():

        column_type = v
        range_str = str(ranges[k]) if ranges[k] else ''
        c.execute(f'INSERT INTO column_types VALUES (?, ?, ?, ?)', (name, k, column_type, range_str))

    conn.commit()
    conn.close()
    try:
        df = pd.read_csv(f'static/{name}.csv')
    except:
        df = pd.read_csv(f'static/{name}.xlsx')

    try:
        mapping_order=pd.read_csv(f'static/{name}order.csv')
    except:
        mapping_order=None        
    #print(df)
    # preprocess each column based on column type and range stored in database
    for column in df.columns:
        column_type = get_column_type(column, name)
        if column_type == 'nominal':
            df[column] = pd.Categorical(df[column]).codes
            print("nominal")
        elif column_type == 'ordinal':
            #if(mapping_order):
            #print(mapping_order)
            df = convert_ordinal(df,column, mapping_order)
            #else:
            #return jsonify({"error": f"No mapping order for dataset {dataset_name}"})    
            
        #print(df)
    df.to_csv(f"static/{name}_processed.csv",index=False)

    print("orderfile",order)
    if(request.files['column_order'].filename):
        print("order")
        file_path1 = os.path.join("static", name+"order.csv")
        order.save(file_path1)
    
    ground_truth_file = request.files['ground_truth']
    if(ground_truth_file.filename):
        print("ground_truth")
        if ground_truth_file.filename.endswith('.txt'):
            ground_truth_df = pd.read_csv(ground_truth_file, header=None, delimiter='\t')
            ground_truth_df.to_csv(f'static/{name}_groundtruth.csv',index=False,header=False)
        elif ground_truth_file.filename.endswith('.csv'):
            ground_truth_df = pd.read_csv(ground_truth_file, header=None)
            ground_truth_df.to_csv(f'static/{name}_groundtruth.csv',index=False,header=False)
        else:
            # handle unsupported file format error
            return "Unsupported file format"
    return 'File uploaded successfully'

if(__name__=="__main__"):
    app.run(host="0.0.0.0",port=5000)
