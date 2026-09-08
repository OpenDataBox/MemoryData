package com.linkpoket.source.domain.aisearch.event;

/**
 * 링크 본문 텍스트 추출을 비동기로 요청하는 이벤트입니다.
 * 링크 저장/수정 트랜잭션 커밋 후 발행되며, 리스너가 URL을 방문해 본문을 추출하여
 * ai_item_body 테이블에 저장합니다.
 *
 * @param linkId 대상 링크 식별자
 * @param url    본문을 추출할 URL
 */
public record LinkBodyExtractionEvent(Long linkId, String url) {
}
