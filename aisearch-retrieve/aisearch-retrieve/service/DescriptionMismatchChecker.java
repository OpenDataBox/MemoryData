package com.linkpoket.source.domain.aisearch.service;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 최근 저장된 아이템 스니펫과 기존의 폴더 요약(description) 내용을 LLM에 대조 전달하여, 
 * 폴더 요약을 재생성(rebuild)해야 할지 예/아니오 JSON으로 답변받아 판정하는 서비스 클래스입니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class DescriptionMismatchChecker {

	private final FolderLatestItemReader latestItemReader;
	private final DescriptionLlmCaller llmCaller;
	private final ObjectMapper objectMapper;

	private static final String SYSTEM_PROMPT =
		"당신은 판단 전문가다. 입력으로 [기존 요약 설명문]과 [신규 추가된 단건 아이템 정보]가 전달된다.\n" +
		"이 신규 아이템의 명칭이나 핵심 설명이 기존 요약문의 범주에 이미 포함되어 자연스럽게 수용될 수 있는지 비교하라.\n" +
		"기존 요약문이 현재 폴더를 대표하기에 어긋남이 없고 충분하다면 needs_rebuild를 false로, " +
		"신규 아이템의 주제가 아주 새롭거나 기존 요약의 범주를 크게 벗어나 요약글을 다시 써야만 정합성이 맞다면 needs_rebuild를 true로 판별하라.\n" +
		"출력은 반드시 json 포맷으로 {\"needs_rebuild\": true/false} 형태로만 응답하라.";

	/**
	 * 기존 요약 설명문과 폴더 내 최근 적재 아이템의 범주 일치 여부를 대조 판정합니다.
	 *
	 * @param folderId           검사 대상 폴더 식별자
	 * @param currentDescription 기존 보관 중이던 요약 설명문
	 * @return 요약 갱신(재생성)이 요구되는 경우 true, 불필요한 경우 false
	 */
	public boolean isMismatch(final Long folderId, final String currentDescription) {
		final String latestItemText = latestItemReader.readLatestItemTitleAndSnippet(folderId);
		if (latestItemText == null || latestItemText.isBlank()) {
			return false;
		}

		final String userPrompt = String.format("[기존 요약 설명문]: %s\n[신규 추가된 단건 아이템 정보]: %s",
			currentDescription, latestItemText);

		try {
			final String rawJson = llmCaller.callJsonYesNo(SYSTEM_PROMPT, userPrompt);
			final JsonNode root = objectMapper.readTree(rawJson);
			return root.path("needs_rebuild").asBoolean(false);
		} catch (final Exception e) {
			log.warn("[AiSearch] Failed to check description mismatch for folderId={}. fallback to true.", folderId, e);
			return true;
		}
	}
}