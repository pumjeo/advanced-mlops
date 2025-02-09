import os
from typing import List, Optional, Tuple

import joblib
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sqlalchemy import create_engine, text

from utils.dates import DateValues

# .env 파일 로드
load_dotenv()

feature_store_url = os.getenv("FEATURE_STORE_URL")
artifacts_path = os.getenv("ARTIFACTS_PATH")

TARGET_NAME = "credit_score"


class Preprocessor:
    """전처리 클래스

    Args:
        model_name (str): 모델명
            해당 이름으로 아티팩트 폴더 아래 관련 객체들이 저장됩니다.
        base_dt (str, optional): 해당 값이 없는 경우 오늘 날짜로 대체됩니다.
    """

    __ROBUST_SCALING_FEATURES = [
        "age",
        "annual_income",
        "monthly_inhand_salary",
        "num_bank_accounts",
        "num_credit_card",
        "interest_rate",
        "num_of_loan",
        "delay_from_due_date",
        "num_of_delayed_payment",
        "changed_credit_limit",
        "num_credit_inquiries",
        "outstanding_debt",
        "credit_utilization_ratio",
        "credit_history_age",
        "total_emi_per_month",
        "amount_invested_monthly",
        "monthly_balance",
    ]
    __TARGET_ENCODING_FEATURES = [
        "occupation",
        "type_of_loan",
        "credit_mix",
        "payment_behaviour",
        "payment_of_min_amount",
    ]

    def __init__(
        self,
        model_name: str,
        base_dt: str = DateValues.get_current_date(),
    ):
        self._model_name = model_name
        self._base_dt = base_dt
        self._save_path = os.path.join(
            artifacts_path,
            "preprocessing",
            self._model_name,
            self._base_dt,
        )
        self._encoder_path = os.path.join(self._save_path, "encoders")
        self._make_dirs()

    def transform(self):
        data = self._fetch_data()
        x_train, y_train, x_test, y_test = self._train_test_split(data=data)
        x_train, x_test = self._transform_with_robust_scaler(
            x_train=x_train, x_test=x_test
        )
        x_train, x_test = self._transform_with_target_encoder(
            x_train=x_train, y_train=y_train, x_test=x_test
        )
        self._save_preprocessed_data(
            feature=x_train, target=y_train, is_train=True
        )
        self._save_preprocessed_data(
            feature=x_test, target=y_test, is_train=False
        )

    def _fetch_data(self) -> pd.DataFrame:
        """`Preprocessor`를 초기화할 때 입력값으로 받은 `base_dt`를 기준으로 데이터를 불러옵니다.
        데이터가 한 건도 존재하지 않는 경우 `ValueError`가 발생합니다.

        Raises:
            ValueError: 데이터가 한 건도 없을 때 발생

        Returns:
            pd.DataFrame: 불러온 데이터
        """

        engine = create_engine(feature_store_url)

        q = f"""
            select *
            from mlops.credit_score_features_target
            where base_dt = '{self._base_dt}'
        """

        with engine.connect() as conn:
            data = pd.read_sql(text(q), con=conn)

        if data.empty:
            raise ValueError("Fetched data is empty! :(")

        return data

    def _train_test_split(
        self,
        data: pd.DataFrame,
        test_size: Optional[float] = 0.3,
        random_state: Optional[int] = 42,
    ) -> Tuple[pd.DataFrame, npt.NDArray, pd.DataFrame, npt.NDArray]:
        """데이터를 학습/테스트 그리고 피처, 타겟으로 나눕니다.

        Args:
            data (pd.DataFrame): 데이터
            test_size (Optional[float], optional): 테스트 데이터 비율
                Defaults to 0.3.
            random_state (Optional[int], optional): 랜덤 시드
                Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, npt.NDArray, pd.DataFrame, npt.NDArray]:
                학습 피처, 학습 타겟, 테스트 피처, 테스트 타겟
        """
        train, test = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        x_train = train.drop([TARGET_NAME], axis=1)
        y_train = train[TARGET_NAME].to_numpy()

        x_test = test.drop([TARGET_NAME], axis=1)
        y_test = test[TARGET_NAME].to_numpy()

        return x_train, y_train, x_test, y_test

    def _transform_with_robust_scaler(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        features: Optional[List[str]] = __ROBUST_SCALING_FEATURES,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """RobustScaler를 이용해서 수치형 변수를 스케일링합니다.

        Args:
            data (pd.DataFrame): 데이터
            features (Optional[List[str]], optional): 대상 피처
                Defaults to `self.__ROBUST_SCALING_FEATURES`.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 스케일링 완료 후 데이터 (학습, 테스트)
        """
        robust_scalers = {}

        for feature in features:
            scaler = RobustScaler()
            robust_scalers[feature] = scaler.fit(x_train[[feature]])
            x_train[feature] = scaler.transform(x_train[[feature]])
            x_test[feature] = scaler.transform(x_test[[feature]])
            print(f"RobustScaler has been applied to {feature}.")

        joblib.dump(
            robust_scalers,
            os.path.join(self._encoder_path, "robust_scaler.joblib"),
        )

        return x_train, x_test

    def _transform_with_target_encoder(
        self,
        x_train: pd.DataFrame,
        y_train: npt.NDArray,
        x_test: pd.DataFrame,
        features: Optional[List[str]] = __TARGET_ENCODING_FEATURES,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """타겟 인코더로 범주형 변수를 인코딩합니다.

        Args:
            x_train (pd.DataFrame): 학습 데이터 피처
            y_train (npt.NDArray): 학습 데이터 타겟값
            x_test (pd.DataFrame): 테스트 데이터 피처
            features (Optional[List[str]], optional): 대상 피처
                Defaults to `self.__TARGET_ENCODING_FEATURES`.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 인코딩 완료 후 데이터 (학습, 테스트)
        """
        target_encoders = {}

        # 모든 피처에 대해서 타겟 인코딩을 적용하여 리스트에 저장한 후 마지막에 합침
        encoded_train_features = []
        encoded_test_features = []

        for feature in features:
            encoder = TargetEncoder(
                categories="auto", target_type="multiclass", random_state=42
            )
            encoder.set_output(transform="pandas")

            target_encoders[feature] = encoder.fit(x_train[[feature]], y_train)

            train_encoded = encoder.transform(x_train[[feature]])
            test_encoded = encoder.transform(x_test[[feature]])

            encoded_train_features.append(train_encoded)
            encoded_test_features.append(test_encoded)

        print("TargetEncoder has been applied to categorical features.")

        joblib.dump(
            target_encoders,
            os.path.join(self._encoder_path, "target_encoder.joblib"),
        )

        # 기존 데이터와 인코딩한 피처를 합침
        # 기존 변수 데이터는 제거
        x_train = self._rename_columns_to_lowercase(
            pd.concat(
                [x_train.drop(columns=features)] + encoded_train_features,
                axis=1,
            )
        )
        x_test = self._rename_columns_to_lowercase(
            pd.concat(
                [x_test.drop(columns=features)] + encoded_test_features, axis=1
            )
        )

        return x_train, x_test

    def _make_dirs(self) -> None:
        """저장될 경로가 존재하지 않으면 해당 폴더를 생성합니다."""
        if not os.path.isdir(self._encoder_path):
            os.makedirs(self._encoder_path)

    def _save_preprocessed_data(
        self,
        feature: pd.DataFrame,
        target: npt.NDArray,
        is_train: Optional[bool] = True,
    ) -> None:
        """전처리된 데이터를 저장합니다.
        입력으로 받은 피처와 타겟값을 다시 합치고, 학습/테스트 데이터에 따라 다른 이름으로 저장합니다.

        Args:
            feature (pd.DataFrame): 피처 데이터
            target (npt.NDArray): 타겟값
            is_train (Optional[bool], optional): 학습 데이터 여부
                Defaults to True.
        """
        file_name = f"{self._model_name}_{'train' if is_train else 'test'}.csv"
        data = feature.copy()
        data[TARGET_NAME] = target

        data.to_csv(os.path.join(self._save_path, file_name), index=False)

    @staticmethod
    def _rename_columns_to_lowercase(data: pd.DataFrame) -> pd.DataFrame:
        """컬럼명을 소문자로 변환한 후 해당 데이터프레임을 반환합니다.

        Args:
            data (pd.DataFrame): 원본 데이터

        Returns:
            pd.DataFrame: 컬럼명을 소문자로 변환한 데이터
        """

        data.columns = [col.lower() for col in data.columns]

        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="An argument parser for preprocessor."
    )

    parser.add_argument(
        "--model_name", type=str, default="credit_score_classification"
    )
    parser.add_argument(
        "--base_dt", type=str, default=DateValues.get_current_date()
    )

    args = parser.parse_args()

    preprocessor = Preprocessor(
        model_name=args.model_name, base_dt=args.base_dt
    )
    preprocessor.transform()
