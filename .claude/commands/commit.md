다음 단계로 staged/unstaged 변경사항을 논리적 단위로 나눠서 커밋해줘.

1. `git status`와 `git diff HEAD`로 전체 변경사항 파악
2. 변경사항을 논리적으로 그룹핑 (예: 새 파일 추가, 기존 파일 수정, 삭제, 변경 내용 등 성격이 다르면 분리)
3. 각 그룹별로:
   - 해당 파일만 `git add`
   - 변경 내용을 반영한 간결한 커밋 메시지 작성
   - 커밋 실행
4. 모든 커밋 완료 후 `git log --oneline -10`으로 결과 확인

커밋 메시지 규칙:
- 영어 사용
- 50자 이내의 제목
- Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com> 를 본문에 미포함
