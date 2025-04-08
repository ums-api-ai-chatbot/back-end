ci/cd 파이프라인 설명입니다!


o CI

o github + git action 사용
github 주소 : https://github.com/ums-api-ai-chatbot/back-end

1. git action trigger 조건 : main 브랜치에 push || pr
2. git action 이 소스코드를 docker image로 말아서 image push 진행
3. image push 하면서 해당 이미지에 태그를 할당
4. 할당된 태그로 deployment.yaml 에 선언된 사용 이미지의 태그 변경

o CD

o kubernetes(k3s) + argocd 사용

서버 주소(kt cloud vm) : 211.254.213.18
argo cd 주소 : http://211.254.213.18:30518
vue 페이지 주소 (front-end) : http://211.254.213.18:32767
fastapi 페이지 주소 (back-end) : http://211.254.213.18:30000

1. k3s로 서버의 전체 pod를 관리합니다.
2. argo cd pod가 git repo의 deployment.yaml을 바라보며 vue ns의 pod 및 fastapi ns의 pod 를 관리 합니다.
3. ci 4번으로 소스코드가 변경되면 argo가 관리하는 pod의 이미지(기존)와 새로운 소스코드의 이미지의 태그가 달라지게 됩니다.
4. argo cd 는 달라진 이미지 태그를 감지하여 out of sync 상태가 됩니다.
5. argo cd 에서 sync를 맟춰주면 관리하는 pod 에서 새로운 이미지를 pull  받아 배포됩니다.
