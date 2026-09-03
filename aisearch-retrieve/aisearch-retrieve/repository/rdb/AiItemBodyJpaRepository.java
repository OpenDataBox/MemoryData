package com.linkpoket.source.domain.aisearch.repository.rdb;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.linkpoket.source.domain.aisearch.entity.AiItemBody;

/**
 * AiItemBody JPA 리포지토리 인터페이스입니다.
 */
public interface AiItemBodyJpaRepository extends JpaRepository<AiItemBody, Long> {

	Optional<AiItemBody> findByItemTypeAndItemId(final String itemType, final Long itemId);

	List<AiItemBody> findAllByItemIdIn(final List<Long> itemIds);
}
