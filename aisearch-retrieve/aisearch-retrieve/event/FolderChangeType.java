package com.linkpoket.source.domain.aisearch.event;

/**
 * 폴더 요약(description) 비동기 조건부 갱신 대상 행위의 종류를 표시하는 Enum 클래스입니다.
 */
public enum FolderChangeType {
	/** 아이템이 새로 등록되었음을 지칭합니다. */
	CREATED,
	/** 아이템의 내부 내용이나 상세가 수정 편집되었음을 지칭합니다. */
	UPDATED,
	/** 아이템이 삭제되었음을 지칭합니다. */
	DELETED
}