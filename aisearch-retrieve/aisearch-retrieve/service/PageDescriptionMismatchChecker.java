package com.linkpoket.source.domain.aisearch.service;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;
import com.linkpoket.source.domain.aisearch.repository.AiFolderMetaDataAccess;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 갱신된 하위 폴더의 요약 텍스트와 기존의 페이지 요약설명 정보를 대조하여,
 * 페이지 설명 요약글을 재생성해야 할지 LLM yes/no 판별을 수행하는 서비스 클래스입니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class PageDescriptionMismatchChecker {

	private final AiFolderMetaDataAccess folderMetaDataAccess;
	private final DescriptionLlmCaller llmCaller;
	private final ObjectMapper objectMapper;

	private static final String SYSTEM_PROMPT =
		"당신은 문서 정합성 판별기다. 입력으로 [기존 페이지 요약문]과 [최근 요약이 변경된 하위 폴더 설명문]이 전달된다.\n" +
		"하위 폴더에 발생한 변경이나 설명 주제가 기존 페이지 요약문의 범주 안에 자연스럽게 스며드는지 확인하라.\n" +
		"요약문을 고치지 않아도 페이지만의 넓은 설명 범주에 완전히 속한다면 needs_rebuild를 false로, " +
		"해당 폴더의 신규 정보로 인해 기존 페이지 요약 설명문을 새롭게 갱신해야만 정합성이 완벽히 유지된다면 needs_rebuild를 true로 판별하라.\n" +
		"출력은 반드시 json 포맷으로 {\"needs_rebuild\": true/false} 형태로만 응답하라.";

	/**
	 * 기존 페이지 요약글과 변경이 감지된 특정 폴더의 요약글을 대조 판정합니다.
	 *
	 * @param pageId             대상 페이지 식별자
	 * @param currentDescription 현재 보관 중인 페이지 요약글
	 * @param triggeredFolderId  갱신을 초래한 하위 폴더 식별자
	 * @return 페이지 요약 갱신(재생성)이 필요하다면 true, 유지 가능하다면 false
	 */
	public boolean isMismatch(final Long pageId, final String currentDescription, final Long triggeredFolderId) {
		final AiFolderMeta folderMeta = folderMetaDataAccess.findByFolderId(triggeredFolderId).orElse(null);
		final String folderDesc = folderMeta != null ? folderMeta.getDescription() : "";
		if (folderDesc == null || folderDesc.isBlank()) {
			return false;
		}

		final String userPrompt = String.format("[기존 페이지 요약문]: %s\n[최근 요약이 변경된 하위 폴더 설명문]: %s",
			currentDescription, folderDesc);

		try {
			final String rawJson = llmCaller.callJsonYesNo(SYSTEM_PROMPT, userPrompt);
			final JsonNode root = objectMapper.readTree(rawJson);
			return root.path("needs_rebuild").asBoolean(false);
		} catch (final Exception e) {
			log.warn("[AiSearch] Failed to check page description mismatch. pageId={}. fallback to true.", pageId, e);
			return true;
		}
	}
}