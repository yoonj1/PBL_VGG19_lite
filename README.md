딥보이스는 딥러닝과 음성 합성 기술을 결합하여 생성된 가짜 음성을 의미한다. 해당 기술은 음성 복원 등의 긍정적 측면을 지니나, 특정인 사칭, 보이스 피싱 등의 범죄 행위에 악용될 수 있다. 따라서 본 프로젝트에서는 딥보이스를 판별할 수 있는 온디바이스 애플리케이션을 제안한다.

 기존의 Deep Voice Detection 모델들은 대규모 음성 데이터를 학습하여 Deep voice와 Real voice를 구분하는 이진 분류를 수행한다. 이러한 모델들은 90% 부근의 높은 정확도를 보이고 있다. 하지만 프로그램이 시스템의 메모리를 과도하게 점유하므로 하드웨어 내에서 구동되지 못 하고 서버에 의존해야 하는 한계점이 있다. 서버 의존 방식은 서버 통신에 따른 시간 지연, 네트워크 연결성 문제를 유발한다.

본 프로젝트는 이러한 문제점을 해결하고자 On-Device AI를 개발하였다. 특히 On-Device AI는 기존의 모델들처럼 클라우드에 사용자의 개인정보를 전송하지 않고 장치 내에서 수행된다. 즉 기기에 개인 정보를 남기지 않아 데이터 프라이버시를 보호할 수 있다.