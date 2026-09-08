package com.linkpoket.source.domain.aisearch.repository;

import java.util.List;
import java.util.Optional;

import com.linkpoket.source.domain.aisearch.entity.AiItemBody;

/**
 * AiItemBody 데이터 접근 인터페이스입니다.
 */
public interface AiItemBodyDataAccess {

	/**
	 * 아이템 타입과 아이템 ID로 본문을 조회합니다.
	 *
	 * @param itemType 아이템 타입
	 * @param itemId 아이템 식별자
	 * @return AiItemBody 객체 (없으면 Optional.empty())
	 */
	Optional<AiItemBody> findByItemTypeAndItemId(final String itemType, final Long itemId);

	/**
	 * 여러 아이템 식별자 목록으로 본문을 일괄 조회합니다.
	 *
	 * @param itemIds 아이템 식별자 목록
	 * @return AiItemBody 객체 목록
	 */
	List<AiItemBody> findAllByItemIdIn(final List<Long> itemIds);

	/**
	 * AiItemBody 데이터를 저장합니다.
	 *
	 * @param body 저장할 본문 엔티티
	 * @return 저장된 AiItemBody 객체
	 */
	AiItemBody save(final AiItemBody body);
}
