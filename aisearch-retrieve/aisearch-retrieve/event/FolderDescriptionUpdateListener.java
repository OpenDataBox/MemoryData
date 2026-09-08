package com.linkpoket.source.domain.aisearch.event;

import java.util.Optional;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.event.EventListener;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.orm.ObjectOptimisticLockingFailureException;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;
import com.linkpoket.source.domain.aisearch.entity.AiFolderMeta;
import com.linkpoket.source.domain.aisearch.error.AiSearchErrorCode;
import com.linkpoket.source.domain.aisearch.repository.AiFolderMetaDataAccess;
import com.linkpoket.source.domain.aisearch.service.DescriptionMismatchChecker;
import com.linkpoket.source.domain.aisearch.service.FolderDescriptionBuilder;
import com.linkpoket.source.domain.folder.repository.FolderDataAccess;
import com.linkpoket.source.domain.folder.entity.Folder;
import com.linkpoket.source.global.config.AsyncConfig;
import com.linkpoket.source.global.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 아이템 소속 폴더 변경 이벤트(FolderItemChangedEvent)를 수신하여, 비동기 스레드에서
 * 폴더 요약 설명을 조건부로 갱신하고 상위 페이지 갱신 이벤트를 전파하는 비동기 이벤트 리스너 클래스입니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class FolderDescriptionUpdateListener {

	private final AiFolderMetaDataAccess folderMetaDataAccess;
	private final DescriptionMismatchChecker mismatchChecker;
	private final FolderDescriptionBuilder descriptionBuilder;
	private final FolderDataAccess folderDataAccess;
	private final ApplicationEventPublisher eventPublisher;

	/**
	 * 아이템 변경 이벤트를 실시간 비동기로 수신하여, 변경 종류가 삭제인 경우 비어있음 여부를 체크하고,
	 * 그 외의 경우 요약 텍스트 재생성 여부를 판별하여 재생성 및 상위 전파를 호출합니다.
	 *
	 * @param event 폴더 식별자와 행위 타입을 갖는 변경 이벤트 객체
	 */
	@Async(AsyncConfig.AI_SEARCH_TASK_EXECUTOR)
	@EventListener
	@Transactional
	public void handleItemChanged(final FolderItemChangedEvent event) {
		if (event.changeType() == FolderChangeType.DELETED) {
			handleDeletion(event.folderId());
			return;
		}
		handleCreationOrUpdate(event.folderId());
	}

	/**
	 * 아이템 등록 및 편집 시 기존 요약글 존재여부 및 주제 일치여부를 체크하여 조건부 재생성을 호출하고,
	 * 소속 페이지 ID가 있는 경우 페이지 설명 검증 연쇄 이벤트를 발행합니다.
	 *
	 * @param folderId 대상 폴더 식별자
	 */
	private void handleCreationOrUpdate(final Long folderId) {
		final AiFolderMeta meta = folderMetaDataAccess.findByFolderId(folderId).orElse(null);
		final String currentDescription = meta != null ? meta.getDescription() : null;

		final boolean needsRebuild;
		if (currentDescription == null || currentDescription.isBlank()) {
			needsRebuild = true;
		} else {
			needsRebuild = mismatchChecker.isMismatch(folderId, currentDescription);
		}

		if (!needsRebuild) {
			log.debug("[AiSearch] folder description kept as-is. folderId={}", folderId);
			return;
		}

		final String rebuilt = descriptionBuilder.build(folderId);
		upsertFolderDescription(folderId, rebuilt);
		log.info("[AiSearch] folder description rebuilt. folderId={}", folderId);

		folderDataAccess.findByFolderId(folderId).ifPresent(folder -> {
			if (folder.getPageId() != null) {
				eventPublisher.publishEvent(new PageDescriptionCheckEvent(folder.getPageId(), folderId));
			}
		});
	}

	/**
	 * 아이템 삭제에 따라 폴더 내부가 비어 있는지 검사하고, 비어 있는 경우 빈 설명("")을 조건부 동기화 기록합니다.
	 *
	 * @param folderId 대상 폴더 식별자
	 */
	private void handleDeletion(final Long folderId) {
		final boolean isEmpty = descriptionBuilder.isFolderEmpty(folderId);
		if (isEmpty) {
			upsertFolderDescription(folderId, "");
			log.info("[AiSearch] folder became empty, description cleared. folderId={}", folderId);
		}
	}

	/**
	 * 데이터베이스에 폴더 요약 정보를 저장합니다. 
	 * 동시성 레이스 컨디션 상황(DataIntegrityViolationException 등)을 안전하게 포획 및 자동 재시도 복구합니다.
	 *
	 * @param folderId  폴더 식별자
	 * @param rebuilt   갱신할 요약 텍스트 설명
	 */
	private void upsertFolderDescription(final Long folderId, final String rebuilt) {
		final String safeDescription = rebuilt == null ? "" : rebuilt;
		try {
			final AiFolderMeta meta = folderMetaDataAccess.findByFolderId(folderId)
				.orElseGet(() -> AiFolderMeta.builder()
					.folderId(folderId)
					.description(safeDescription)
					.rawTitleIndex("")
					.build());
			meta.updateDescriptionAndEmbedding(safeDescription, meta.getEmbedding());
			folderMetaDataAccess.saveAndFlush(meta);
		} catch (final DataIntegrityViolationException | ObjectOptimisticLockingFailureException e) {
			final AiFolderMeta meta = folderMetaDataAccess.findByFolderId(folderId)
				.orElseThrow(() -> new BusinessException(AiSearchErrorCode.SEARCH_UNAVAILABLE));
			meta.updateDescriptionAndEmbedding(safeDescription, meta.getEmbedding());
			folderMetaDataAccess.save(meta);
		}
	}
}