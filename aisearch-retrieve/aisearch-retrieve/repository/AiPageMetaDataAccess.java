package com.linkpoket.source.domain.aisearch.repository;

import java.util.Optional;
import com.linkpoket.source.domain.aisearch.entity.AiPageMeta;

/**
 * AI 페이지 요약 메타데이터에 접근하기 위한 데이터 액세스 인터페이스입니다.
 */
public interface AiPageMetaDataAccess {
	/**
	 * 페이지 식별자로 AI 페이지 요약 데이터를 조회합니다.
	 *
	 * @param pageId 페이지 식별자
	 * @return AI 페이지 메타데이터 (없으면 Optional.empty())
	 */
	Optional<AiPageMeta> findByPageId(Long pageId);

	/**
	 * AI 페이지 메타데이터를 저장(영속화)합니다.
	 *
	 * @param meta 영속화할 AI 페이지 메타데이터 객체
	 * @return 저장 완료된 AI 페이지 메타데이터 객체
	 */
	AiPageMeta save(AiPageMeta meta);

	/**
	 * AI 페이지 메타데이터를 즉시 데이터베이스와 동기화(flush)하여 저장합니다.
	 *
	 * @param meta 동기화할 AI 페이지 메타데이터 객체
	 */
	void saveAndFlush(AiPageMeta meta);
}