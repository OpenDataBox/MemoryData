package com.linkpoket.source.domain.aisearch.event;

/**
 * 특정 폴더 내부의 아이템 구성에 변동이 일어났을 때, 요약 설명 갱신을 판단하도록 발행하는 이벤트 record 객체입니다.
 *
 * @param folderId   대상의 폴더 식별자
 * @param changeType 아이템 등록, 수정, 삭제 등의 변경 행위 타입
 */
public record FolderItemChangedEvent(Long folderId, FolderChangeType changeType) {
}