package com.linkpoket.source.domain.aisearch.repository.adapter;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Repository;

import com.linkpoket.source.domain.aisearch.entity.AiItemBody;
import com.linkpoket.source.domain.aisearch.repository.AiItemBodyDataAccess;
import com.linkpoket.source.domain.aisearch.repository.rdb.AiItemBodyJpaRepository;

import lombok.RequiredArgsConstructor;

/**
 * AiItemBodyDataAccess 구현 클래스입니다.
 */
@Repository
@RequiredArgsConstructor
class AiItemBodyDataAccessImpl implements AiItemBodyDataAccess {

	private final AiItemBodyJpaRepository jpaRepository;

	@Override
	public Optional<AiItemBody> findByItemTypeAndItemId(final String itemType, final Long itemId) {
		return jpaRepository.findByItemTypeAndItemId(itemType, itemId);
	}

	@Override
	public List<AiItemBody> findAllByItemIdIn(final List<Long> itemIds) {
		return jpaRepository.findAllByItemIdIn(itemIds);
	}

	@Override
	public AiItemBody save(final AiItemBody body) {
		return jpaRepository.save(body);
	}
}
