import mlflow
import mlflow.pyfunc


class MlflowRegistryClient:
    def __init__(self):
        if mlflow.active_run():
            self.run = mlflow.active_run()
            self.run_info = self.run.info
            self.artifact_uri = self.run_info.artifact_uri
            self.run_id = self.run_info.run_id

        self.client = mlflow.tracking.MlflowClient()

    @staticmethod
    def log_model(*args, **kwargs):
        # sugar, only for completeness of this tool
        mlflow.pyfunc.log_model(*args, **kwargs)

    @staticmethod
    def _filter_tags(model_versions: mlflow.store.entities.paged_list.PagedList, tags: dict = {}) -> list:
        # if tags are specified then only models with the same (subset of) tags are returned
        model_info_tags = []
        if tags:
            for mv in model_versions:
                for k, v in mv.tags.items():
                    if tags.get(k, None) == v:
                        model_info_tags.append(mv)
            return model_info_tags
        else:
            return list(model_versions)

    def register_model(self, model_name: str) -> mlflow.entities.model_registry.ModelVersion:
        # registers model previously stored via `log_model` during the current run
        model_uri = f"runs:/{self.run_id}/"
        mv = mlflow.register_model(model_uri, model_name)
        return mv

    def set_model_version_tags(self, model_name: str, model_version, tags: dict):
        # if tags are specified then only models with the same (subset of) tags are selected
        for k, v in tags.items():
            self.client.set_model_version_tag(model_name, model_version, k, v)

    def get_model_info(self, model_name: str) -> mlflow.store.entities.paged_list.PagedList:
        # get list of all registerd versions info for specified model
        return self.client.search_model_versions(f"name='{model_name}'")

    def get_model_info_tags(self, model_name: str, tags: dict = {}) -> list:
        # get list of all registerd versions info for specified model with given tags
        model_versions = self.get_model_info(model_name)
        return self._filter_tags(model_versions, tags)

    def get_model_version_info(self, model_name: str, model_version: str) -> mlflow.entities.model_registry.ModelVersion:
        # get info for specified version of certain model
        model_info = self.get_model_info(model_name)
        return [mv for mv in model_info if mv.version == model_version][0]

    def get_model_info_latest(self, model_name: str) -> mlflow.entities.model_registry.ModelVersion:
        # get last recorded version info for specified model
        return self.get_model_info(model_name)[-1]

    def get_model_info_stage(
        self, model_name: str, model_stage: str, tags: dict = {}, take_latest: bool = True
    ) -> mlflow.entities.model_registry.ModelVersion:
        # get version info about model in specified stage with specified tags
        # If there are multiple versions fulfilling the criterie, either the latest model is returned,
        # or all the models are returned, depending on the `take_latest` param.
        mv = self.get_model_info(model_name)
        model_info_base = [v for v in mv if v.current_stage == model_stage]
        model_info = self._filter_tags(model_info_base, tags)
        # take the latest model in case of multiple models fulfilling the criteria
        if model_info:
            if take_latest:
                return model_info[-1]
            else:
                return model_info
        else:
            return

    def get_model_by_version(self, model_name, version) -> mlflow.pyfunc.PyFuncModel:
        dl_uri = self.client.get_model_version_download_uri(model_name, version)
        return mlflow.pyfunc.load_model(model_uri=dl_uri + f'/{model_name}')

    def get_model_by_stage(self, model_name: str, model_stage: str, tags: dict = {}) -> mlflow.pyfunc.PyFuncModel:
        stage_version = self.get_model_info_stage(model_name, model_stage, tags).version
        return self.get_model_by_version(model_name, stage_version)

    def model_stage_transition(self, model_name: str, version: str, target_stage: str):
        # move model version to target_stage
        errmsg = f"Unsupported model stage! '{target_stage}'"
        assert target_stage in ["None", "Staging", "Production", "Archived"], errmsg
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage
        )

    def full_model_transition(self, model_name: str, tags: dict = {}):
        # move model from Production stage to Archived
        # move model from Staging stage to Production
        prod_version = self.get_model_info_stage(model_name, "Production", tags).version
        stage_version = self.get_model_info_stage(model_name, "Staging", tags).version
        self.model_stage_transition(model_name, prod_version, "Archived")
        self.model_stage_transition(model_name, stage_version, "Production")
