package com.linkpoket.source.domain.aisearch.service;

/**
 * LLM 호출 시 함께 전달할 이전 대화 한 턴(질문 또는 답변)을 표현하는 값 객체입니다.
 *
 * @param role    메시지 발행 주체 ("user" 또는 "assistant")
 * @param content 메시지 본문
 */
public record ConversationTurn(String role, String content) {

	/**
	 * 사용자 질문 턴을 생성합니다.
	 *
	 * @param content 질문 본문
	 * @return 사용자 역할의 대화 턴
	 */
	public static ConversationTurn user(final String content) {
		return new ConversationTurn("user", content);
	}

	/**
	 * 어시스턴트 답변 턴을 생성합니다.
	 *
	 * @param content 답변 본문
	 * @return 어시스턴트 역할의 대화 턴
	 */
	public static ConversationTurn assistant(final String content) {
		return new ConversationTurn("assistant", content);
	}
}
