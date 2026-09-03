package com.linkpoket.source.domain.aisearch.repository;

import java.util.List;
import java.util.Optional;
import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;

/**
 * AiFolderMeta 데이터 접근 인터페이스입니다.
 */
public interface AiFolderMetaDataAccess {
	/**
	 * 폴더 ID로 메타데이터를 조회합니다.
	 *
	 * @param folderId 폴더 식별자
	 * @return AiFolderMeta 객체 (없으면 Optional.empty())
	 */
	Optional<AiFolderMeta> findByFolderId(final Long folderId);

	/**
	 * AiFolderMeta 데이터를 저장합니다.
	 *
	 * @param meta 저장할 메타데이터
	 * @return 저장된 AiFolderMeta 객체
	 */
	AiFolderMeta save(final AiFolderMeta meta);

	/**
	 * AiFolderMeta 데이터를 데이터베이스에 즉시 동기화하여 저장(flush)합니다.
	 *
	 * @param meta 동기화할 AiFolderMeta 객체
	 */
	void saveAndFlush(final AiFolderMeta meta);

	/**
	 * 여러 폴더 식별자 목록으로 요약 메타데이터 목록을 일괄 조회합니다.
	 *
	 * @param folderIds 폴더 식별자 목록
	 * @return 폴더 요약 메타데이터 목록
	 */
	List<AiFolderMeta> findAllByFolderIdIn(final List<Long> folderIds);
}
