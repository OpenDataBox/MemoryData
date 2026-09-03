package com.linkpoket.source.domain.aisearch.repository.rdb;

import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import com.linkpoket.source.domain.aisearch.entity.AiPageMeta;

/**
 * Spring Data JPA를 기반으로 하는 AiPageMeta 레포지토리 인터페이스입니다.
 */
public interface AiPageMetaJpaRepository extends JpaRepository<AiPageMeta, Long> {
	/**
	 * 페이지 식별자로 AI 페이지 요약 데이터를 조회합니다.
	 *
	 * @param pageId 페이지 식별자
	 * @return AI 페이지 메타데이터 (없으면 Optional.empty())
	 */
	Optional<AiPageMeta> findByPageId(Long pageId);
}