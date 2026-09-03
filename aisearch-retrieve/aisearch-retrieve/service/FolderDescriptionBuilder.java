package com.linkpoket.source.domain.aisearch.service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import com.linkpoket.source.domain.link.repository.LinkDataAccess;
import com.linkpoket.source.domain.link.entity.Link;
import com.linkpoket.source.domain.note.repository.NoteDataAccess;
import com.linkpoket.source.domain.note.entity.Note;
import com.linkpoket.source.domain.file.repository.FileDataAccess;
import com.linkpoket.source.domain.file.entity.File;
import com.linkpoket.source.domain.postit.repository.FolderPostItDataAccess;
import lombok.RequiredArgsConstructor;

/**
 * 특정 폴더 내부의 모든 원시 구성 아이템(링크 제목, 노트 내용, 파일 명칭, 포스트잇 내용 등) 목록을
 * 정렬 수집하여 LLM에게 요약본 작성을 요청 및 빌드해내는 서비스 클래스입니다.
 */
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class FolderDescriptionBuilder {

	private final LinkDataAccess linkDataAccess;
	private final NoteDataAccess noteDataAccess;
	private final FileDataAccess fileDataAccess;
	private final FolderPostItDataAccess folderPostItDataAccess;
	private final DescriptionLlmCaller llmCaller;

	private static final int MAX_EXTRACT_ITEMS = 30;

	private static final String SYSTEM_PROMPT =
		"당신은 폴더 정보 요약 전문가다. 제시되는 여러 개별 아이템들의 명칭, 링크 URL, 노트 컨텐츠 등을 기반으로\n" +
		"이 폴더가 어떤 목적과 범주의 정보들을 통합하여 관리하고 있는지 설명하는 1~2문장의 간결한 요약문을 작성하라.\n" +
		"특정 아이템에 치우치지 않는 범용적이고 핵심적인 설명문이어야 하며, 한국어로 200자 이내로 정중히 대답하라.";

	/**
	 * 대상 폴더의 모든 아이템들을 긁어 모아 LLM을 통해 한글 요약 설명을 새로이 빌드(생성)합니다.
	 *
	 * @param folderId 요약 대상 폴더 식별자
	 * @return 새로 조립 생성된 요약 설명 문자열
	 */
	public String build(final Long folderId) {
		final List<Long> ids = List.of(folderId);
		final List<String> items = new ArrayList<>();

		linkDataAccess.findAllByFolderIdIn(ids).stream()
			.limit(MAX_EXTRACT_ITEMS)
			.forEach(l -> items.add("- 링크아이템: " + l.getName() + " (" + l.getUrl() + ")"));

		noteDataAccess.findAllByFolderIdIn(ids).stream()
			.limit(MAX_EXTRACT_ITEMS)
			.forEach(n -> {
				final String body = n.getContent() != null ? n.getContent() : "";
				final String snippet = body.length() <= 30 ? body : body.substring(0, 30);
				items.add("- 노트아이템: " + n.getName() + " / 내용: " + snippet);
			});

		fileDataAccess.findFileEntitiesByFolderIds(ids).stream()
			.limit(MAX_EXTRACT_ITEMS)
			.forEach(f -> items.add("- 파일아이템: " + f.getName()));

		folderPostItDataAccess.findAllByFolderIdIn(ids).stream()
			.limit(MAX_EXTRACT_ITEMS)
			.forEach(p -> items.add("- 포스트잇: " + p.getContent()));

		if (items.isEmpty()) {
			return "";
		}

		final String userPrompt = items.stream().collect(Collectors.joining("\n"));
		return llmCaller.callPlainText(SYSTEM_PROMPT, userPrompt);
	}

	/**
	 * 대상 폴더에 현재 하위 아이템(링크, 노트, 파일, 포스트잇)이 전혀 존재하지 않는 비어 있는 상태인지 판별합니다.
	 *
	 * @param folderId 대상 폴더 식별자
	 * @return 비어 있다면 true, 구성원이 존재한다면 false
	 */
	public boolean isFolderEmpty(final Long folderId) {
		final List<Long> ids = List.of(folderId);
		final boolean noLinks = linkDataAccess.findAllByFolderIdIn(ids).isEmpty();
		final boolean noNotes = noteDataAccess.findAllByFolderIdIn(ids).isEmpty();
		final boolean noFiles = fileDataAccess.findFileEntitiesByFolderIds(ids).isEmpty();
		final boolean noPostIts = folderPostItDataAccess.findAllByFolderIdIn(ids).isEmpty();
		return noLinks && noNotes && noFiles && noPostIts;
	}
}