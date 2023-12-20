FROM python:3.9
WORKDIR /opt/source-code/
COPY . /opt/source-code/
RUN pip install Flask
RUN pip install requests
RUN pip install joblib
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install Jinja2
RUN pip install requests
RUN pip install pandas
RUN pip install numpy
CMD [ "python", "app.py"]