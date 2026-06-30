from flask import Flask, request, jsonify
from recommender import FoodRecommender
from utils import setup_logging, validate_glucose_payload, error_response, success_response
from config import Config

logger = setup_logging()

app = Flask(__name__)

# Initialize recommender once at startup - hard fail if Excel missing/invalid
try:
    recommender = FoodRecommender(
        excel_path=Config.EXCEL_PATH,
        sensitivity=Config.DEFAULT_SENSITIVITY
    )
except Exception as e:
    logger.error(f"❌ Failed to initialize FoodRecommender: {e}")
    raise


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "foods_loaded": len(recommender.df),
        "sensitivity": recommender.sensitivity
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True, silent=True)
        validated = validate_glucose_payload(data)

        result = recommender.recommend(
            risk=validated["risk"],
            predictions=validated["predictions"],
            uncertainty=validated["uncertainty"],
            current_glucose=validated["current_glucose"],
            carbs_limit=validated["carbs_limit"],
            meal_type=validated["meal_type"]
        )

        payload, status = success_response(result)
        return jsonify(payload), status

    except ValueError as e:
        payload, status = error_response(str(e), 400)
        return jsonify(payload), status

    except Exception as e:
        logger.exception("Unhandled error in /recommend")
        payload, status = error_response(f"Internal server error: {e}", 500)
        return jsonify(payload), status


@app.route("/foods/search", methods=["GET"])
def search_food():
    query = request.args.get("q", "")
    if not query:
        payload, status = error_response("Query param 'q' is required", 400)
        return jsonify(payload), status

    results = recommender.search_food(query)
    payload, status = success_response({"foods": results, "count": len(results)})
    return jsonify(payload), status


@app.route("/foods/gi-band/<band>", methods=["GET"])
def foods_by_gi_band(band):
    if band.lower() not in ("low", "medium", "high"):
        payload, status = error_response("band must be one of: low, medium, high", 400)
        return jsonify(payload), status

    results = recommender.get_food_by_gi_band(band)
    payload, status = success_response({"foods": results, "count": len(results)})
    return jsonify(payload), status


@app.route("/foods", methods=["GET"])
def all_foods():
    limit = request.args.get("limit", 50, type=int)
    results = recommender.get_all_foods(limit=limit)
    payload, status = success_response({"foods": results, "count": len(results)})
    return jsonify(payload), status


@app.route("/sensitivity", methods=["POST"])
def set_sensitivity():
    try:
        data = request.get_json(force=True, silent=True) or {}
        sensitivity = float(data.get("sensitivity"))
        recommender.set_sensitivity(sensitivity)
        payload, status = success_response({"sensitivity": recommender.sensitivity})
        return jsonify(payload), status
    except (TypeError, ValueError) as e:
        payload, status = error_response(str(e), 400)
        return jsonify(payload), status


@app.errorhandler(404)
def not_found(e):
    payload, status = error_response("Endpoint not found", 404)
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
