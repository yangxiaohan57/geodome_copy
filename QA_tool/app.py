import os
from datetime import datetime
import numpy as np
import pandas as pd
from glob import glob
from flask import Flask, render_template, url_for, flash, request, session
from flask_session import Session

DEBUG = True
app = Flask(__name__)
SESSION_TYPE = 'filesystem'

app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'something unique and secret.'
Session(app)

@app.route('/reset')
def reset():
    session["idx"]=0
    return render_template("main.html", fname=fnames[0],
                               image="{}/{}.png".format(img_dst, fnames[0]),
                               raster="{}/{}.png".format(raster_dst, fnames[0]))

img_dst = "./static/_imgs"
raster_dst = "./static/_rasters"
fname_csv = 'fnames.csv'


if not os.path.exists("./static/{}".format(fname_csv)):
    img_list = glob("{}/*.png".format(img_dst))
    raster_list = glob("{}/*_raster.png".format(raster_dst))

    fnames = [os.path.basename(p).split('.')[0] for p in img_list+raster_list]
    fnames = list(set([fn for fn in fnames if not (fn.endswith('_N') or fn.endswith('_raster'))]))
    fname_df = pd.DataFrame(data=fnames, columns=['fname'])
    fname_df['Decision'] = np.nan
else:
    fname_df = pd.read_csv("./static/{}".format(fname_csv))
    fnames = list(fname_df[fname_df.Decision.isna()]['fname'])

@app.route('/', methods=['GET', 'POST'])
def main():
    idx = session.get("idx", 0)
    if(idx==len(fnames)-1):
        flash('No more images')

    if request.method == 'POST':
        if request.form['btn'] == 'Save':
            fname_df.to_csv('./static/{}'.format(fname_csv), index=False)
            flash('Progress Saved at {}'.format(datetime.now().strftime("%H:%M:%S")))

        else:
            fname_df.loc[(fname_df.fname == fnames[idx]).values, ['Decision']] = request.form['btn']
            if(idx<len(fnames)-1):
                idx +=1
                session["idx"] = idx
            print(idx)
        return render_template("main.html", fname=fnames[idx],
                            image="{}/{}.png".format(img_dst, fnames[idx]),
                            raster="{}/{}_raster.png".format(raster_dst, fnames[idx]))

    elif request.method == 'GET':
        return render_template("main.html", fname=fnames[idx],
                               image="{}/{}.png".format(img_dst, fnames[idx]),
                               raster="{}/{}_raster.png".format(raster_dst, fnames[idx]))


if __name__ == '__main__':
    app.run()
