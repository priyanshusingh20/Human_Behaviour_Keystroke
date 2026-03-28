from ml_model import MultiUserModel

multi_model = MultiUserModel()
multi_model.train()
from flask import Flask, render_template, request
from Verifier import Verifier
from flask import session, redirect, url_for
import json
import os

app = Flask(__name__)
app.secret_key = "super_secret_key"
USERS = {
    "priyanshu": "12345678",
    "rahul": "abcd",
    "ashwini": "1234"
}

multi_model = MultiUserModel()
multi_model.train()

@app.route('/', methods=['GET', 'POST'])
def login():

    result = ""
    confidence = 0

    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            typed = request.form.get('typed', '').strip()
            keystrokes = json.loads(request.form.get('keystrokes', '[]'))

            # -------------------------
            # STEP 1: BASIC VALIDATION
            # -------------------------
            if not username or not password:
                return render_template('index.html', result="Enter all fields")

            # -------------------------
            # STEP 2: PASSWORD CHECK
            # -------------------------
            if username not in USERS or USERS[username] != password:
                return render_template('index.html', result="Invalid Password")

            # -------------------------
            # STEP 3: KEYSTROKE CHECK
            # -------------------------
            if typed != "helloworld":
                return render_template('index.html', result="Type correctly")

            if len(keystrokes) < 3:
                return render_template('index.html', result="Typing too short")

            # Convert to probe
            probe_data = {}

            for i, k in enumerate(keystrokes):
                hold = float(k.get("hold_time", 0)) / 100
                pp = float(k.get("press_press", 0)) / 100
                rp = float(k.get("release_press", 0)) / 100

                probe_data[str(i)] = {
                    "hold_1": hold,
                    "press_press": pp,
                    "release_press": rp,
                    "release_release": rp,
                    "hold_2": hold,
                    "total_time": hold + pp,
                    "slope_h1": hold,
                    "slope_pp": pp,
                    "slope_rp": rp,
                    "slope_rr": rp,
                    "slope_h2": hold,
                    "slope_tt": hold + pp
                }

            # -------------------------
            # STEP 4: VERIFY USER
            # -------------------------
            verifier = Verifier(username, typed)
            decision, nb, l2, confidence = verifier.compare_metrics(probe_data)

            # -------------------------
            # STEP 5: IDENTIFY USER
            # -------------------------
            detected_user, detect_conf = multi_model.predict_user(probe_data)

            print("Entered:", username)
            print("Detected:", detected_user)

            # -------------------------
            # FINAL SECURITY CHECK 🔐
            # -------------------------
            if decision and detected_user == username:

                session['user'] = username

                return render_template('dashboard.html',
                                       user=username,
                                       confidence=confidence)

            else:
                return render_template('index.html',
                                       result=f"Access Denied | Detected: {detected_user}",
                                       confidence=confidence)

        except Exception as e:
            return render_template('index.html',
                                   result=f"Error: {str(e)}")

    return render_template('index.html')

@app.route('/enroll', methods=['POST'])
def enroll():
    try:
        username = request.form.get('username')
        typed = request.form.get('typed')
        keystrokes = json.loads(request.form.get('keystrokes', '[]'))

        probe_data = {}

        for i, k in enumerate(keystrokes):
            hold = float(k.get("hold_time", 0))
            pp = float(k.get("press_press", 0))
            rp = float(k.get("release_press", 0))

            probe_data[str(i)] = {
                "hold_1": hold,
                "press_press": pp,
                "release_press": rp,
                "release_release": rp,
                "hold_2": hold,
                "total_time": hold + pp,
                "slope_h1": hold / 10 if hold else 0,
                "slope_pp": pp / 10 if pp else 0,
                "slope_rp": rp / 10 if rp else 0,
                "slope_rr": rp / 10 if rp else 0,
                "slope_h2": hold / 10 if hold else 0,
                "slope_tt": (hold + pp) / 10 if hold else 0
            }

        word = "-".join(typed.split(" "))
        file_path = f'dataset/password/{username}_{word}.json'

        if os.path.exists(file_path):
            with open(file_path) as f:
                data = json.load(f)
        else:
            data = []

        data.append(probe_data)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        return "Enrollment successful"

    except Exception as e:
        return str(e)

@app.route('/dataset')
def dataset():
    username = request.args.get('username')
    typed = request.args.get('typed')

    file_path = f'dataset/password/{username}_{typed}.json'

    if not os.path.exists(file_path):
        return []

    with open(file_path) as f:
        data = json.load(f)

    return data

@app.route('/graph', methods=['POST'])
def graph():
    keystrokes = json.loads(request.form['keystrokes'])

    times = [k['press_press'] for k in keystrokes]
    keys = [k['key'] for k in keystrokes]

    return {
        "times": times,
        "keys": keys
    }

@app.route('/identify', methods=['POST'])
def identify():
    try:
        keystrokes = json.loads(request.form['keystrokes'])

        print("Incoming keystrokes:", keystrokes)
        print("Type:", type(keystrokes))

        #  Convert list → dict (VERY IMPORTANT)
        probe_data = {}

        for i, k in enumerate(keystrokes):
            hold = float(k.get("hold_time", 0)) / 100
            pp = float(k.get("press_press", 0)) / 100
            rp = float(k.get("release_press", 0)) / 100

            probe_data[str(i)] = {
                "hold_1": hold,
                "press_press": pp,
                "release_press": rp,
                "release_release": rp,
                "hold_2": hold,
                "total_time": hold + pp,
                "slope_h1": hold,
                "slope_pp": pp,
                "slope_rp": rp,
                "slope_rr": rp,
                "slope_h2": hold,
                "slope_tt": hold + pp
            }

        print("Converted probe:", probe_data)

        #  Now correct format
        user, confidence = multi_model.predict_user(probe_data)

        return {
            "user": user,
            "confidence": confidence
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "user": "Error",
            "confidence": 0
        }
    
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')
    
if __name__ == "__main__":    
    app.run(debug=True)