package com.linkpoket.source.domain.aisearch.error;

import org.springframework.http.HttpStatus;

import com.linkpoket.source.global.error.code.spec.ErrorCode;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * AI 검색 도메인 에러코드입니다.
 */
@Getter
@RequiredArgsConstructor
public enum AiSearchErrorCode implements ErrorCode {

	SEARCH_TIMEOUT(HttpStatus.GATEWAY_TIMEOUT, "ai.search.error.timeout"),
	UNAUTHORIZED_PAGE_ACCESS(HttpStatus.FORBIDDEN, "ai.search.error.unauthorized_page"),
	SEARCH_UNAVAILABLE(HttpStatus.SERVICE_UNAVAILABLE, "ai.search.error.unavailable"),
	QUOTA_EXCEEDED(HttpStatus.TOO_MANY_REQUESTS, "ai.search.error.quota_exceeded"),
	CONVERSATION_NOT_FOUND(HttpStatus.NOT_FOUND, "ai.search.error.conversation_not_found"),
	CANARY_NOT_ALLOWED(HttpStatus.FORBIDDEN, "ai.search.error.canary_not_allowed");

	private final HttpStatus httpStatus;
	private final String errorMessage;

	@Override
	public HttpStatus getHttpStatus() {
		return httpStatus;
	}

	@Override
	public String getErrorMessage() {
		return errorMessage;
	}
}
