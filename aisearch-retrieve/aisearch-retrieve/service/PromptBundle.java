package com.linkpoket.source.domain.aisearch.service;

import java.util.List;

/**
 * 2단계 LLM 질문 탐색용 폴더 프롬프트 문자열과 탐색 후보 폴더 ID 목록을 묶어 관리하는 record 객체입니다.
 *
 * @param folderListPrompt 마크다운 형태의 폴더 리스트 설명 프롬프트
 * @param folderIds        검색 후보 폴더 식별자 목록
 */
public record PromptBundle(String folderListPrompt, List<Long> folderIds) {
}