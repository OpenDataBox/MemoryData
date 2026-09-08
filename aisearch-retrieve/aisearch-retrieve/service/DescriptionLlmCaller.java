package com.linkpoket.source.domain.aisearch.service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.linkpoket.source.global.ai.OpenAiSearchApi;
import com.linkpoket.source.domain.aisearch.error.AiSearchErrorCode;
import com.linkpoket.source.domain.aisearch.util.PromptInjectionGuard;
import com.linkpoket.source.global.exception.BusinessException;
import io.github.resilience4j.bulkhead.annotation.Bulkhead;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 폴더 및 페이지의 요약(description) 조건부 갱신 시 활용되는 LLM 비동기 백그라운드 호출 전담 서비스 클래스입니다.
 * 실시간 검색 파이프라인과 자원을 분리하기 위해 openAiSearchBackground 벌크헤드 슬롯을 활용합니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class DescriptionLlmCaller {

	private final OpenAiSearchApi openAiSearchApi;
	private final ObjectMapper objectMapper;

	@Value("${ai-search.openai-api-key:dummy-key}")
	private String apiKey;

	@Value("${ai-search.openai-model:gpt-4o-mini}")
	private String model;

	private static final int MAX_TOKENS_YES_NO = 100;
	private static final int MAX_TOKENS_PLAIN_TEXT = 150;

	/**
	 * 예/아니오 및 판정 정보가 수록될 수 있는 JSON 포맷으로 LLM 호출을 백그라운드에서 진행합니다.
	 *
	 * @param systemPrompt 시스템 기준 지시 프롬프트 (PromptInjectionGuard 래핑 자동 적용)
	 * @param userPrompt   비교 및 매칭용 사용자 데이터 정보
	 * @return 결과 JSON 텍스트 문자열
	 */
	public String callJsonYesNo(final String systemPrompt, final String userPrompt) {
		final Map<String, Object> body = new HashMap<>();
		body.put("model", model);
		body.put("temperature", 0);
		body.put("response_format", Map.of("type", "json_object"));
		body.put("max_tokens", MAX_TOKENS_YES_NO);
		body.put("messages", List.of(
			Map.of("role", "system", "content", PromptInjectionGuard.wrap(systemPrompt)),
			Map.of("role", "user", "content", userPrompt)
		));
		final String raw = callLlmBackground(body);
		final JsonNode root = readTree(raw);
		return root.path("choices").path(0).path("message").path("content").asText("").trim();
	}

	/**
	 * 단문 평문 텍스트 요약 생성을 목적으로 LLM 호출을 백그라운드에서 진행합니다.
	 *
	 * @param systemPrompt 요약 생성 지침 프롬프트 (PromptInjectionGuard 래핑 자동 적용)
	 * @param userPrompt   요약 가공 대상 데이터 정보
	 * @return 생성된 평문 요약 텍스트
	 */
	public String callPlainText(final String systemPrompt, final String userPrompt) {
		final Map<String, Object> body = new HashMap<>();
		body.put("model", model);
		body.put("temperature", 0);
		body.put("max_tokens", MAX_TOKENS_PLAIN_TEXT);
		body.put("messages", List.of(
			Map.of("role", "system", "content", PromptInjectionGuard.wrap(systemPrompt)),
			Map.of("role", "user", "content", userPrompt)
		));
		final String raw = callLlmBackground(body);
		final JsonNode root = readTree(raw);
		return root.path("choices").path(0).path("message").path("content").asText("").trim();
	}

	/**
	 * 백그라운드 서킷 브레이커와 벌크헤드 격리(openAiSearchBackground) 하에서 OpenAI API를 Feign 호출합니다.
	 *
	 * @param body API 요청 Payload 맵 객체
	 * @return API 응답 문자열
	 */
	@Bulkhead(name = "openAiSearchBackground")
	@CircuitBreaker(name = "openAiSearchBackground", fallbackMethod = "callLlmFallback")
	String callLlmBackground(final Map<String, Object> body) {
		return openAiSearchApi.chat("Bearer " + apiKey, body);
	}

	/**
	 * 백그라운드 LLM 호출 도중 서킷오픈 또는 장애가 발생했을 때 작동하는 폴백 메서드입니다.
	 *
	 * @param body      실패한 요청 Payload
	 * @param throwable 예외 정보 객체
	 * @return 예외를 생성하여 바깥으로 전파합니다 (SEARCH_UNAVAILABLE)
	 */
	String callLlmFallback(final Map<String, Object> body, final Throwable throwable) {
		log.error("[AiSearch Description LLM] call failed or circuit open. cause={}", throwable.toString());
		throw new BusinessException(AiSearchErrorCode.SEARCH_UNAVAILABLE);
	}

	/**
	 * 문자열 응답 텍스트 데이터를 JsonNode 구조형 데이터로 파싱합니다.
	 *
	 * @param text JSON 문자열
	 * @return JsonNode 객체
	 */
	private JsonNode readTree(final String text) {
		try {
			return objectMapper.readTree(text);
		} catch (final Exception e) {
			log.warn("[AiSearch Description] JSON parse failed.");
			throw new BusinessException(AiSearchErrorCode.SEARCH_UNAVAILABLE);
		}
	}
}