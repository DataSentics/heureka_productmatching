import os
from mlflow.pyfunc import PythonModel

from utilities.component import process_input
from utilities.attributes import Attributes
from preprocessing.models import exact_match
from xgboostmatching.models.model import XGBoostMatchingModel


class RegistryMatchingModelConstructor(PythonModel):
    # see https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_context(self, context):
        # automatically loads all the artifacts from context in the background
        # not executed when constructed directly
        self.get_matching_model(context)

    def predict(self, context, model_input):
        # model_input = {"product": Product, "offer": Offer}
        # it is not possible to pass product and offer as two separate imputs
        return self.model.predict(model_input)

    def get_matching_model(self, context):
        artifacts = context.artifacts
        os.makedirs('/data', exist_ok=True)
        if artifacts.get('transformer'):
            artifacts['transformer'] = process_input(
                artifacts['transformer'], '/data'
            )

        namesimilarity = exact_match.ExactMatchModel(
            tok_norm_args=f"input_pmi={artifacts.get('pmi')}"
        )

        attributes = None
        if artifacts.get('attributes'):
            attributes = Attributes(
                from_=artifacts.get('attributes'),
            )

        xgboostmodel = XGBoostMatchingModel(
            xgboost_path=artifacts.get('xgb'),
            namesimilarity=namesimilarity,
            attributes=attributes,
            thresholds_path=artifacts.get("thresholds", None),
            tok_norm_args=f"input_pmi={artifacts.get('pmi')}",
            unit_conversions=os.getenv("CONFIG__unit_conversions", True),
            price_reject_a=float(os.getenv("CONFIG__price_reject_a", '1000.0')),
            price_reject_b=float(os.getenv("CONFIG__price_reject_b", '400.0')),
            price_reject_c=float(os.getenv("CONFIG__price_reject_c", '2.5'))
        )
        self.model = xgboostmodel
