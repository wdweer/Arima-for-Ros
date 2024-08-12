import rospy
from geometry_msgs.msg import Twist
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Arima:
    def __init__(self):
        rospy.init_node('Arima_AI_Model')
        self.target_velocity_sub = rospy.Subscriber('/target_velocity', Twist, self.target_velocity_callback)
        self.vx_data = []
        self.vy_data = []
        self.plot_done = False  # 플롯이 한번만 실행되게 제어하는 플래그
        self.rate = rospy.Rate(10)  # 적절한 빈도로 설정
        while not rospy.is_shutdown():
            self.rate.sleep()

    def target_velocity_callback(self, data):
        target_vx = data.linear.x
        target_vy = data.linear.y
        self.vx_data.append(target_vx)
        self.vy_data.append(target_vy)
        print(f"Data length: {len(self.vx_data)}")
        if len(self.vx_data) >= 100 :
            self.run_arima()  # 데이터 초기화
            self.plot_done = True  # 플롯이 실행되었음을 표시

    def run_arima(self):
        try:
            print("Running ARIMA model...")
            self.model = ARIMA(self.vx_data, order=(10, 1, 10))  # p, d, q 값 수정
            self.model_fit = self.model.fit()
            self.forecast = self.model_fit.forecast(steps=10)
            print("Forecast: ", self.forecast)

            # 실제 값과 예측 값을 비교하여 정확성 평가
            if len(self.vx_data) >= 110:  # 최소 110개의 데이터가 필요함 (예측 데이터 + 비교할 실제 데이터)
                actual = self.vx_data[-10:]  # 예측 시점 이후의 실제 데이터
                self.evaluate_forecast(actual, self.forecast)

        except Exception as e:
            print(f"Error in ARIMA model: {e}")

    def evaluate_forecast(self, actual, forecast):
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

if __name__ == '__main__':
    try:
        Arima()
    except rospy.ROSInterruptException:
        pass
