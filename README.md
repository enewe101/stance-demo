Steps to start developping.  Starting from the src dir, do this:

    pip install -r requirements.txt
	cd src
    echo DATA_DIR = "'"`pwd`/data"'" | cat - SETTINGS.py.template > \
        SETTINGS.py
    python prepare_vectorizer.py
