package com.linkpoket.source.domain.aisearch.service;

import java.time.LocalDateTime;

/**
 * 링크, 노트, 파일 등 개별적인 아이템 종류를 통합하여 컴포저 정렬 시 활용하는 내부 공용 뷰 DTO record 객체입니다.
 *
 * @param itemId   아이템 고유 식별자
 * @param itemType 아이템 유형(LINK, NOTE, FILE 등)
 * @param title    아이템 명칭/제목
 * @param url      링크 혹은 파일 접속 URL (노트는 빈 값)
 * @param body     노트 내용 혹은 아이템 본문 내용
 * @param editedAt 최종 수정/업데이트 날짜
 */
public record ItemView(
	Long itemId,
	String itemType,
	String title,
	String url,
	String body,
	LocalDateTime editedAt
) {
}