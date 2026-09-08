package com.linkpoket.source.domain.aisearch.service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import com.linkpoket.source.domain.folder.entity.Folder;
import com.linkpoket.source.domain.folder.repository.FolderDataAccess;
import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;
import com.linkpoket.source.domain.aisearch.repository.AiFolderMetaDataAccess;
import lombok.RequiredArgsConstructor;

/**
 * 특정 페이지 내부의 모든 폴더 요약(description) 정보를 읽어 모아
 * LLM을 통해 페이지 전체의 대표 요약 설명을 새로이 빌드(생성)하는 서비스 클래스입니다.
 */
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class PageDescriptionBuilder {

	private final FolderDataAccess folderDataAccess;
	private final AiFolderMetaDataAccess folderMetaDataAccess;
	private final DescriptionLlmCaller llmCaller;

	private static final String SYSTEM_PROMPT =
		"당신은 페이지 정보 요약 전문가다. 제시되는 페이지 내 하위 폴더들의 요약 설명 정보들을 종합하여\n" +
		"이 페이지 전체가 주로 다루는 지식 정보 및 활용 목적을 대변하는 1~2문장의 간결한 요약문을 작성하라.\n" +
		"각 폴더의 세부 사항에 치우치지 않고 페이지를 한눈에 대표하는 설명문이어야 하며, 한국어로 200자 이내로 정중히 대답하라.";

	/**
	 * 페이지 내 모든 폴더들의 요약 설명문들을 바탕으로 LLM을 호출하여 페이지 요약을 새로이 생성합니다.
	 *
	 * @param pageId 요약 대상 페이지 식별자
	 * @return 새로 조립 생성된 페이지 요약 텍스트
	 */
	public String build(final Long pageId) {
		final List<Folder> folders = new ArrayList<>();
		folders.addAll(folderDataAccess.findTopLevelFolders(pageId));
		final List<Long> parentIds = folders.stream().map(Folder::getId).collect(Collectors.toList());
		if (!parentIds.isEmpty()) {
			folders.addAll(folderDataAccess.findChildFolders(parentIds));
		}

		if (folders.isEmpty()) {
			return "";
		}

		final List<Long> folderIds = folders.stream().map(Folder::getId).collect(Collectors.toList());
		final List<AiFolderMeta> metas = folderMetaDataAccess.findAllByFolderIdIn(folderIds);

		final String mergedDescriptions = metas.stream()
			.map(AiFolderMeta::getDescription)
			.filter(desc -> desc != null && !desc.isBlank())
			.map(desc -> "- 폴더 요약: " + desc)
			.collect(Collectors.joining("\n"));

		if (mergedDescriptions.isEmpty()) {
			return "";
		}

		return llmCaller.callPlainText(SYSTEM_PROMPT, mergedDescriptions);
	}
}