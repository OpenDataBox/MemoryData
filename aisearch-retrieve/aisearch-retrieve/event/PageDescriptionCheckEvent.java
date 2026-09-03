package com.linkpoket.source.domain.aisearch.event;

/**
 * 특정 페이지의 하위 구조물 변경에 따라 연쇄적인 페이지 설명 요약 업데이트를 판단 검사하도록 트리거하는 이벤트 Record 객체입니다.
 *
 * @param pageId   대상 페이지 식별자
 * @param folderId 원인을 제공한 폴더 식별자
 */
public record PageDescriptionCheckEvent(Long pageId, Long folderId) {
}