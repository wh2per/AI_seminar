# Day1

## 1. Machine Learning이란?
  * ### Supervised Learning(지도학습)
    - 미리 레이블링 된 준비된 데이터를 가지고 학습을 하는 방법
    - Training data set : Labeled data
    - 예 : AlphaGo
  * ### Unsupervised Learning(비지도학습)
    - 일일이 레이블링 할 수 없는 데이터를 이용해 할습할 때
    - 예 : google news grouping
    
## 2. Supervised Learning(지도학습)
  * ### Regression(회기)
    - 연속성이 있는 범위에서의 결과를 예측하는 기계학습
    - 예 : 시험 점수 예측, 거리에 따른 배달시간 예측
  * ### Binary classification(이진분류)
    - 예측할 class가 두 가지인 경우
    - 예 : 시험 결과가 Pass일 것인가 Non-pass일 것인가 
  * ### Multi-label classification
    - 예측할 class가 여러가지인 경우
    - 예 : 시험 걸과에 의한 학점이 A, B, C, D, F 중 무엇이 될 것인가
    
## 3. Linear Regression
  * H(x) = Wx + b 형태를 띔
  * cost = (H(x) - Y)<sup>2</sup>의 값들의 평균 = <img src="Day1/img/cost1.JPG" width="200" height="50">
  * 궁극적인 목표는 가장 작은 cost값을 갖도록 하는 W와 b를 구하는 것
  * cost함수의 그래프  
    <br><img src="Day1/img/cost2.JPG" width="400" height="350"> 
  * W값을 찾는 방법 
    <br><img src="Day1/img/cost3.JPG" width="400" height="330"> 
  * 더 많은 변수가 있는 경우 = <img src="Day1/img/cost4.JPG" width="250" height="30"> 
  
## 4. Logistic classification
  * 0또는 1의 결과를 가지는 Binary Classification일 때 Linear Regression의 문제점
    - 큰 Input 값으로 인해 H(x)를 변형할 경우 틀린 결과 값(Y)가 도출 될 수 있음
    - H(x)를 변형하지 않고, x에 큰 값을 입력하면 1보다 큰 결과
  * _Sigmoid Function_(Logistic Function)의 등장으로 문제점 해결!
  * <img src="Day1/img/sigmoid1.JPG" width="450" height="290"> 
  * Logistic classification의 H(x)를 cost함수에 적용하면 최적의 cost를 찾을 수 없음
  * cost함수는 예측 값(H(x))과 결과값(Y)이 일치할수록 0에 가까워지도록 log를 이용하여 수정하자
  * <img src="Day1/img/sigmoid2.JPG" width="400" height="250"> 
  
## 5. Softmax Regression
  * 여러개의 class가 있을때 그것을 예측하기 위한 multinomial classification
  * <img src="Day1/img/softmax1.JPG" width="300" height="250"> 
  * 각 레이블마다의 결과를 확률값으로 변경
    <br><img src="Day1/img/softmax2.JPG" width="300" height="250">
  * 변형된 cost 함수
    <br><img src="Day1/img/softmax3.JPG" width="370" height="130">
    <br><img src="Day1/img/softmax4.JPG" width="320" height="70">

## 6. Overfitting
  * 학습데이터에만 너무 맞도록 학습이 된 경우 -> 배운 것만 알고 다른 것들은 모르는 케이스
    <br><img src="Day1/img/overfitting1.JPG" width="380" height="200">
  * Regularization(일반화)를 이용하여 해결
    <br><img src="Day1/img/overfitting2.JPG" width="440" height="220">
