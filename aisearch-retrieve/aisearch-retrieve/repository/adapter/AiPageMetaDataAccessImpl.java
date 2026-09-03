package com.linkpoket.source.domain.aisearch.repository.adapter;

import java.util.Optional;
import org.springframework.stereotype.Repository;
import com.linkpoket.source.domain.aisearch.entity.AiPageMeta;
import com.linkpoket.source.domain.aisearch.repository.AiPageMetaDataAccess;
import com.linkpoket.source.domain.aisearch.repository.rdb.AiPageMetaJpaRepository;
import lombok.RequiredArgsConstructor;

/**
 * AiPageMetaDataAccess 인터페이스의 RDB 구현체 어댑터 클래스입니다.
 */
@Repository
@RequiredArgsConstructor
public class AiPageMetaDataAccessImpl implements AiPageMetaDataAccess {

	private final AiPageMetaJpaRepository jpaRepository;

	/**
	 * JPA 레포지토리를 호출하여 페이지 식별자에 매칭되는 AI 페이지 메타데이터를 조회합니다.
	 *
	 * @param pageId 페이지 식별자
	 * @return AI 페이지 메타데이터 (없으면 Optional.empty())
	 */
	@Override
	public Optional<AiPageMeta> findByPageId(final Long pageId) {
		return jpaRepository.findByPageId(pageId);
	}

	/**
	 * JPA 레포지토리를 통해 AI 페이지 메타데이터를 저장 및 영속화합니다.
	 *
	 * @param meta 영속화할 AI 페이지 메타데이터 객체
	 * @return 저장 완료된 AI 페이지 메타데이터 객체
	 */
	@Override
	public AiPageMeta save(final AiPageMeta meta) {
		return jpaRepository.save(meta);
	}

	/**
	 * JPA 레포지토리를 통해 AI 페이지 메타데이터를 즉시 RDB에 동기화하여 저장(flush)합니다.
	 *
	 * @param meta 즉시 동기화하여 저장할 AI 페이지 메타데이터 객체
	 */
	@Override
	public void saveAndFlush(final AiPageMeta meta) {
		jpaRepository.saveAndFlush(meta);
	}
}