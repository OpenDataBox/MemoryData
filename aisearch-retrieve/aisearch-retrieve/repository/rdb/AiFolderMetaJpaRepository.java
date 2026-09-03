package com.linkpoket.source.domain.aisearch.repository.rdb;

import org.springframework.data.jpa.repository.JpaRepository;

import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;

/**
 * AiFolderMeta JPA 리포지토리 인터페이스입니다.
 */
public interface AiFolderMetaJpaRepository extends JpaRepository<AiFolderMeta, Long> {
}
