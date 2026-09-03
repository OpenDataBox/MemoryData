package com.linkpoket.source.domain.aisearch.service;

import java.time.LocalDateTime;

/**
 * 폴더 내 개별 아이템들의 최신 수정시각 대조 및 식별 스니펫 관리를 담당하는 DTO record 객체입니다.
 *
 * @param title    아이템 명칭/제목
 * @param detail   아이템 상세/내용/URL 등
 * @param editedAt 최종 수정 시각
 */
public record ItemSummary(String title, String detail, LocalDateTime editedAt) {
}