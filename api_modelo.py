import os
import logging
import datetime
import jwt
import joblib
import numpy as np

from functools import wraps
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# Authentication and JWT settings
JWT_SECRET = "TestPassw0rd"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

# logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB settings
DB_URL = "sqlite:///predictions.db"
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


# Create the database tables
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(engine)

# Load the model
model = joblib.load("iris_model.pkl")
logger.info("Model loaded successfully.")

# Initialize Flask app
app = Flask(__name__)
predictions_cache = {}

# Authentication example
TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpassword"


def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)

    return decorated


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token}), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 401


@app.route("/predict", methods=["POST"])
@token_required
def predict():
    """
    Endpoint protegido por token  para obter previsões do modelo.
    Corpo (JSON):
    {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
    }
    """
    data = request.get_json(force=True)
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing input data: {e}")
        return jsonify({"error": "Invalid input data"}), 400

    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        logger.info("Cache hit para %s", features)
        predicted_class = predictions_cache[features]

    else:
        # Realiza a previsão
        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
       
        # Armazenar em cache
        predictions_cache[features] = predicted_class
        logger.info("Cache updated para %s", features)

    # Armazenar no banco de dados
    db = SessionLocal()
    new_prediction = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class,
    )
    db.add(new_prediction)
    db.commit()
    db.close()
    logger.info("Prediction saved to database: %s", new_prediction)
    return jsonify({"predicted_class": predicted_class}), 200


@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    """
    Endpoint para listar previsões armazenadas no banco de dados.
    Parametros opcionais (via query string):
       - limit: (int): quantidade máxima de previsões a retornar
       - offset: (int): número de previsões a pular antes de retornar os resultados
    Exemplo:
        /predictions?limit=10&offset=0
    """

    limit = request.args.get("limit", default=10, type=int)
    offset = request.args.get("offset", default=0, type=int)

    db = SessionLocal()
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    db.close()

    results = []

    for pred in predictions:
        results.append(
            {
                "id": pred.id,
                "sepal_length": pred.sepal_length,
                "sepal_width": pred.sepal_width,
                "petal_length": pred.petal_length,
                "petal_width": pred.petal_width,
                "predicted_class": pred.predicted_class,
                "created_at": pred.created_at.isoformat(),
            }
        )

    return jsonify(results), 200

if __name__ == "__main__":
    app.run(debug=True)