package com.linkpoket.source.domain.aisearch.repository.adapter;

import java.util.List;
import java.util.Optional;
import org.springframework.stereotype.Repository;
import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;
import com.linkpoket.source.domain.aisearch.repository.AiFolderMetaDataAccess;
import com.linkpoket.source.domain.aisearch.repository.rdb.AiFolderMetaJpaRepository;
import lombok.RequiredArgsConstructor;

/**
 * AiFolderMetaDataAccess 구현 클래스입니다.
 */
@RequiredArgsConstructor
@Repository
public class AiFolderMetaDataAccessImpl implements AiFolderMetaDataAccess {

	private final AiFolderMetaJpaRepository jpaRepository;

	/**
	 * JPA 레포지토리 기본 제공 findById를 호출하여 폴더 식별자에 매칭되는 AI 폴더 메타데이터를 조회합니다.
	 *
	 * @param folderId 폴더 식별자
	 * @return AI 폴더 메타데이터 (없으면 Optional.empty())
	 */
	@Override
	public Optional<AiFolderMeta> findByFolderId(final Long folderId) {
		return jpaRepository.findById(folderId);
	}

	/**
	 * JPA 레포지토리 save를 통해 AI 폴더 메타데이터를 저장 및 영속화합니다.
	 *
	 * @param meta 저장할 AI 폴더 메타데이터 객체
	 * @return 저장 완료된 AI 폴더 메타데이터 객체
	 */
	@Override
	public AiFolderMeta save(final AiFolderMeta meta) {
		return jpaRepository.save(meta);
	}

	/**
	 * JPA 레포지토리 saveAndFlush를 호출하여 AI 폴더 메타데이터를 즉시 데이터베이스에 플러시 동기화 저장합니다.
	 *
	 * @param meta 즉시 저장할 AI 폴더 메타데이터 객체
	 */
	@Override
	public void saveAndFlush(final AiFolderMeta meta) {
		jpaRepository.saveAndFlush(meta);
	}

	/**
	 * JPA 레포지토리 findAllById를 호출하여 복수의 폴더 식별자 목록에 일치하는 AI 폴더 메타데이터 목록을 벌크 일괄 조회합니다.
	 *
	 * @param folderIds 폴더 식별자 목록
	 * @return 조회된 AI 폴더 메타데이터 목록
	 */
	@Override
	public List<AiFolderMeta> findAllByFolderIdIn(final List<Long> folderIds) {
		return jpaRepository.findAllById(folderIds);
	}
}
